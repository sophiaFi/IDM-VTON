"""
Preprocessing script for the FIT dataset.

Generates agnostic masks and densepose maps for all target images and saves them
into {data_root}/agnostic-mask/ and {data_root}/image-densepose/ respectively.

Run from the repo root:
    python preprocess_fit.py --data_root data_mini_test [--device cuda]
"""

import argparse
import json
import os
import sys

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
        target_path = os.path.join(data_root, record["target"])
        target_stem = os.path.splitext(os.path.basename(record["target"]))[0]
        target_ext  = os.path.splitext(os.path.basename(record["target"]))[1]

        mask_out  = os.path.join(out_mask_dir, f"{target_stem}_mask.png")
        pose_out  = os.path.join(out_pose_dir, f"{target_stem}{target_ext}")

        already_done = os.path.exists(mask_out) and os.path.exists(pose_out)
        if already_done:
            continue

        target_pil = Image.open(target_path).convert("RGB")

        # OpenPose + Parsing both expect 384×512
        small = target_pil.resize((384, 512))

        # --- Agnostic mask ---
        if not os.path.exists(mask_out):
            try:
                keypoints = openpose_model(small)
                model_parse, _ = parsing_model(small)
                mask, _ = get_mask_location("hd", category, model_parse, keypoints)
                mask.save(mask_out)
            except Exception as exc:
                print(f"\n[WARN] mask failed for {record['target']}: {exc}")
                skipped.append(("mask", record["target"]))

        # --- DensePose ---
        if not os.path.exists(pose_out):
            try:
                pose_img = run_densepose(densepose_predictor, densepose_context, small)
                pose_img.save(pose_out)
            except Exception as exc:
                print(f"\n[WARN] densepose failed for {record['target']}: {exc}")
                skipped.append(("densepose", record["target"]))

    if skipped:
        print(f"\nSkipped {len(skipped)} item(s):")
        for kind, path in skipped:
            print(f"  [{kind}] {path}")
    else:
        print("\nAll images processed successfully.")


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
        help="Device for DensePose and OpenPose (default: cuda if available)",
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0,
        help="GPU id passed to Parsing and OpenPose constructors (default: 0)",
    )
    args = parser.parse_args()

    preprocess(
        data_root=os.path.abspath(args.data_root),
        category=args.category,
        device=args.device,
        gpu_id=args.gpu_id,
    )


if __name__ == "__main__":
    main()
