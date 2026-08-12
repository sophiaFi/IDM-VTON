"""
Preprocessing script for the FIT dataset.

Steps (all idempotent — already-computed outputs are skipped):
  1. Agnostic masks  → {data_root}/agnostic-mask/
  2. DensePose maps  → {data_root}/image-densepose/
  3. Garment captions (BLIP-2) → written back into measurements.json as
     "garment_caption" on each record.

Run from the repo root:
    python preprocess_fit.py --data_root data_mini_test [--device cuda]
    python preprocess_fit.py --data_root data_mini_test --captions-only
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO_ROOT, "gradio_demo"))

from detectron2.data.detection_utils import _apply_exif_orientation, convert_PIL_to_numpy
from detectron2.engine.defaults import DefaultPredictor

import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from utils_mask import get_mask_location


# ---------------------------------------------------------------------------
# DensePose helper — build predictor once and reuse across images
# ---------------------------------------------------------------------------

def _build_densepose_predictor(device: str):
    args = apply_net.create_argument_parser().parse_args([
        "show",
        os.path.join(_REPO_ROOT, "configs", "densepose_rcnn_R_50_FPN_s1x.yaml"),
        os.path.join(_REPO_ROOT, "ckpt", "densepose", "model_final_162be9.pkl"),
        "dp_segm", "-v",
        "--opts", "MODEL.DEVICE", device,
    ])
    cfg = apply_net.ShowAction.setup_config(args.cfg, args.model, args, [])
    predictor = DefaultPredictor(cfg)
    context = apply_net.ShowAction.create_context(args, cfg)
    return predictor, context


def run_densepose(predictor, context, pil_image: Image.Image) -> Image.Image:
    """
    Run DensePose on a PIL image (any size) and return the segmentation
    visualisation as a PIL RGB image at the same resolution.
    """
    img_bgr = convert_PIL_to_numpy(
        _apply_exif_orientation(pil_image), format="BGR"
    )
    with torch.no_grad():
        instances = predictor(img_bgr)["instances"]
        vis = apply_net.ShowAction.execute_on_outputs(
            context, {"image": img_bgr}, instances
        )
    # vis is BGR numpy; convert to RGB PIL
    return Image.fromarray(vis[:, :, ::-1])


# ---------------------------------------------------------------------------
# Main preprocessing loop
# ---------------------------------------------------------------------------

def expand_mask_for_garment(
    mask: Image.Image,
    record: dict,
    mask_w: int = 384,
    mask_h: int = 512,
) -> Image.Image:
    """
    Expand an agnostic mask to account for the target garment being larger than
    the input person's current garment.

    The expansion is driven by measurements already in the record:
      - Width  : garment_bust / body_bust  → horizontal dilation
      - Length : garment_length / body_height → vertical extension (bottom only)
      - Sleeves: garment_sleeve_length / body_height → extra arm reach

    All scale factors are clamped to [1, max_*] so we only ever grow the mask,
    never shrink it. A small baseline dilation (5 px) is always applied first to
    avoid leaving a 1-px gap around the input garment's boundary.
    """
    # Ratios of target garment dimensions to body dimensions
    width_ratio  = record["garment_bust"]          / max(record["body_bust"],   1.0)
    length_ratio = record["garment_length"]         / max(record["body_height"], 1.0)
    sleeve_ratio = record["garment_sleeve_length"]  / max(record["body_height"], 1.0)

    # Reference "typical" ratios for a normally-fitting upper garment (empirical).
    # Anything beyond these triggers extra dilation.
    REF_WIDTH_RATIO  = 1.05   # garment bust ≈ 5 % larger than body bust
    REF_LENGTH_RATIO = 0.28   # garment covers ~28 % of body height
    REF_SLEEVE_RATIO = 0.15   # sleeve ≈ 15 % of body height

    # Convert excess ratio to pixel dilation amounts (clamped, never negative)
    H_SCALE = mask_h / 512.0   # normalise to the canonical 512 px height
    W_SCALE = mask_w / 384.0

    extra_width  = max(0.0, width_ratio  - REF_WIDTH_RATIO)  * mask_w * 0.5
    extra_length = max(0.0, length_ratio - REF_LENGTH_RATIO) * mask_h
    extra_sleeve = max(0.0, sleeve_ratio - REF_SLEEVE_RATIO) * mask_h * 0.5

    # Cap expansions at reasonable maximums (in pixels at canonical resolution)
    extra_width  = min(extra_width,  int(40 * W_SCALE))
    extra_length = min(extra_length, int(60 * H_SCALE))
    extra_sleeve = min(extra_sleeve, int(40 * H_SCALE))

    mask_np = np.array(mask.convert("L"))
    mask_np = (mask_np > 127).astype(np.uint8) * 255

    # Step 1: baseline dilation to fill boundary gaps from the input garment
    baseline_px = max(3, int(5 * H_SCALE))
    mask_np = cv2.dilate(mask_np, np.ones((baseline_px, baseline_px), np.uint8))

    # Step 2: horizontal (bust/width) expansion — symmetric left/right
    if extra_width >= 1:
        kw = int(extra_width) * 2 + 1
        mask_np = cv2.dilate(mask_np, np.ones((1, kw), np.uint8))

    # Step 3: downward (length) extension — pad bottom only by shifting the mask
    # up by extra_length pixels and dilating, then take union
    if extra_length >= 1:
        shift = int(extra_length)
        shifted = np.zeros_like(mask_np)
        if shift < mask_h:
            shifted[:mask_h - shift] = mask_np[shift:]
        mask_np = np.maximum(mask_np, shifted)

    # Step 4: sleeve reach — symmetric horizontal + slight downward dilation
    if extra_sleeve >= 1:
        ksl = int(extra_sleeve) * 2 + 1
        mask_np = cv2.dilate(mask_np, np.ones((ksl, ksl), np.uint8))

    mask_np = np.clip(mask_np, 0, 255).astype(np.uint8)
    return Image.fromarray(mask_np)


def preprocess(data_root: str, category: str, device: str, gpu_id: int):
    measurements_path = os.path.join(data_root, "measurements.json")
    with open(measurements_path) as f:
        records = json.load(f)

    out_mask_dir = os.path.join(data_root, "agnostic-mask")
    out_pose_dir = os.path.join(data_root, "image-densepose")
    os.makedirs(out_mask_dir, exist_ok=True)
    os.makedirs(out_pose_dir, exist_ok=True)

    print("Loading models...")
    parsing_model = Parsing(gpu_id)
    openpose_model = OpenPose(gpu_id)
    openpose_model.preprocessor.body_estimation.model.to(device)
    densepose_predictor, densepose_context = _build_densepose_predictor(device)
    print("Models loaded.")

    skipped = []

    for record in tqdm(records, desc="Preprocessing"):
        target_stem = os.path.splitext(os.path.basename(record["target"]))[0]
        target_ext  = os.path.splitext(os.path.basename(record["target"]))[1]

        mask_out  = os.path.join(out_mask_dir, f"{target_stem}{target_ext}")
        pose_out  = os.path.join(out_pose_dir, f"{target_stem}{target_ext}")

        already_done = os.path.exists(mask_out) and os.path.exists(pose_out)
        if already_done:
            continue

        # Use the input person image (wearing a different garment) — not the target —
        # so the mask and densepose match what is available at inference time.
        person_pil = Image.open(os.path.join(data_root, record["person"])).convert("RGB")

        # OpenPose + Parsing both expect 384×512
        small = person_pil.resize((384, 512))

        # --- Agnostic mask ---
        if not os.path.exists(mask_out):
            try:
                keypoints = openpose_model(small)
                model_parse, _ = parsing_model(small)
                mask, _ = get_mask_location("hd", category, model_parse, keypoints)
                mask = expand_mask_for_garment(mask, record, mask_w=384, mask_h=512)
                mask.save(mask_out)
            except Exception as exc:
                print(f"\n[WARN] mask failed for {record['person']}: {exc}")
                skipped.append(("mask", record["person"]))

        # --- DensePose ---
        if not os.path.exists(pose_out):
            try:
                pose_img = run_densepose(densepose_predictor, densepose_context, small)
                pose_img.save(pose_out)
            except Exception as exc:
                print(f"\n[WARN] densepose failed for {record['person']}: {exc}")
                skipped.append(("densepose", record["person"]))

    if skipped:
        print(f"\nSkipped {len(skipped)} item(s):")
        for kind, path in skipped:
            print(f"  [{kind}] {path}")
    else:
        print("\nAll images processed successfully.")


# ---------------------------------------------------------------------------
# BLIP-2 garment captioning
# ---------------------------------------------------------------------------

# Prompt that steers BLIP-2 away from describing the background / mannequin
# and towards the garment itself.
_BLIP2_MODEL = "Salesforce/blip2-opt-2.7b"
# Completion-style prompt: the model fills in the rest of the sentence directly,
# avoiding conversational openers like "It's a..." or "This is a...".
_BLIP2_PROMPT = "a upper garment photo of exactly a"
_FALLBACK_PROMPT = "a garment that is"

_STRIP_PREFIXES = (
    "it's a ", "it is a ", "this is a ", "this is an ",
    "the garment is a ", "the garment is an ", "the image shows a ",
    "it's an ", "it is an ",
)


def generate_captions(data_root: str, device: str, batch_size: int = 8):
    """
    Run BLIP-2 on every unique cloth image in measurements.json and write the
    resulting caption back as a "garment_caption" field on each record.

    Already-captioned records (where "garment_caption" is non-empty) are
    skipped so the step is safe to re-run after an interruption.
    """
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    measurements_path = os.path.join(data_root, "measurements.json")
    with open(measurements_path) as f:
        records = json.load(f)

    def _is_valid_caption(cap) -> bool:
        # Reject missing, empty, or captions that still contain the prompt
        # prefix (artefact of an earlier buggy run that decoded the full
        # prompt+answer sequence instead of just the answer).
        if not cap:
            return False
        if "Question:" in cap or "Answer:" in cap:
            return False
        return True

    # Collect cloth paths that still need a caption, deduplicated by path so
    # the same garment image is not captioned twice if it appears in multiple
    # records.
    cloth_to_caption: dict = {}
    needs_caption: list = []
    for record in records:
        cloth_path = record["cloth"]
        existing = record.get("garment_caption", "")
        if _is_valid_caption(existing):
            cloth_to_caption[cloth_path] = existing
        elif cloth_path not in cloth_to_caption:
            cloth_to_caption[cloth_path] = None
            needs_caption.append(cloth_path)

    if not needs_caption:
        print("All records already have captions — nothing to do.")
        return

    print(f"Loading BLIP-2 ({_BLIP2_MODEL}) on {device}…")
    processor = Blip2Processor.from_pretrained(_BLIP2_MODEL, use_fast=False)
    model = Blip2ForConditionalGeneration.from_pretrained(
        _BLIP2_MODEL,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
    ).to(device)
    model.eval()
    print("BLIP-2 loaded.")

    weight_dtype = torch.float16 if "cuda" in device else torch.float32

    def _caption_one(image: Image.Image, prompt: str) -> str:
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {
            k: (v.to(device, dtype=weight_dtype) if v.is_floating_point() else v.to(device))
            for k, v in inputs.items()
        }
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=50)
        text = processor.decode(generated_ids[0], skip_special_tokens=True).strip().lower()
        for prefix in _STRIP_PREFIXES:
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
        return text.strip()

    # Process one image at a time to avoid any batching/padding interaction
    # with BLIP-2's internal inputs_embeds construction.
    empty_paths = []
    for cloth_path in tqdm(needs_caption, desc="Captioning"):
        full_path = os.path.join(data_root, cloth_path)
        image = Image.open(full_path).convert("RGB")

        caption = _caption_one(image, _BLIP2_PROMPT)
        if not caption:
            caption = _caption_one(image, _FALLBACK_PROMPT)
        if not caption:
            print(f"[WARN] empty caption for {cloth_path}")
            empty_paths.append(cloth_path)
        cloth_to_caption[cloth_path] = caption

    if empty_paths:
        print(f"\n{len(empty_paths)} image(s) produced no caption:")
        for p in empty_paths:
            img = Image.open(os.path.join(data_root, p))
            print(f"  {p}  size={img.size}  mode={img.mode}")

    # Write captions back into every record and save
    for record in records:
        record["garment_caption"] = cloth_to_caption[record["cloth"]]

    with open(measurements_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nCaptions written to {measurements_path}")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess FIT dataset images.")
    parser.add_argument(
        "--data_root", required=True,
        help="Path to the FIT dataset root (containing measurements.json)",
    )
    parser.add_argument(
        "--category", default="upper_body",
        choices=["upper_body", "lower_body", "dresses"],
        help="Garment category used when computing the agnostic mask (default: upper_body)",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for DensePose, OpenPose, and BLIP-2 (default: cuda if available)",
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0,
        help="GPU id passed to Parsing and OpenPose constructors (default: 0)",
    )
    parser.add_argument(
        "--captions-only", action="store_true",
        help="Skip mask/densepose generation and only run BLIP-2 captioning.",
    )
    parser.add_argument(
        "--caption-batch-size", type=int, default=8,
        help="Number of garment images per BLIP-2 forward pass (default: 8).",
    )
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)

    if not args.captions_only:
        preprocess(
            data_root=data_root,
            category=args.category,
            device=args.device,
            gpu_id=args.gpu_id,
        )

    generate_captions(
        data_root=data_root,
        device=args.device,
        batch_size=args.caption_batch_size,
    )


if __name__ == "__main__":
    main()
