"""
inference_measurement_adapter.py - IDM-VTON + Measurement Adapter (Extension 3) inference on FIT data.

Without --checkpoint_dir: pretrained yisol/IDM-VTON, measurements are ignored (baseline).

With --checkpoint_dir <path>: loads the base model, overlays conv_in (unet_lora.pt), builds the
MeasurementEncoder + MeasurementAdapter from adapter_config.json and installs the third parallel
cross-attention branch in every TryonNet attn2 layer.

    checkpoint-<step>/
        unet_lora.pt, measurement_adapter.pt, measurement_encoder.pt, adapter_config.json

Optional --measurement_scale multiplies the (trained) measurement gates at inference time
(1.0 = as trained, 0.0 = branch off) for sensitivity analyses.

Example:
    python inference_measurement_adapter.py --data_dir data/data_mini_test \
        --checkpoint_dir output/checkpoint-500 --output_dir result_adapter --num_samples 4
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
from PIL import Image, ImageFilter
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

# Preprocessing models live inside gradio_demo/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gradio_demo"))

from ip_adapter.ip_adapter import Resampler
from src.fit_dataset import FITDatasetWithMeasurements
from src.measurement_adapter import (
    MeasurementAdapter, MeasurementAdapterContext,
    get_hidden_sizes, install_measurement_adapter,
)
from src.measurement_encoder import MeasurementEncoder
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

_MEASUREMENT_MASK_KWARGS = (
    "body_bust_cm", "garment_bust_cm", "garment_length_cm", "body_height_cm", "densepose_torso_mask",
)


def _use_original_idm_vton_mask():
    """Make FITDatasetWithMeasurements use the unmodified upstream get_mask_location
    (gradio_demo/utils_mask_original.py): no measurement-driven expansion and the original dilation sizes."""
    import src.fit_dataset as fit_dataset
    from utils_mask_original import get_mask_location as upstream_get_mask_location

    def original_get_mask_location(*a, **kw):
        for k in _MEASUREMENT_MASK_KWARGS:
            kw.pop(k, None)
        return upstream_get_mask_location(*a, **kw)

    fit_dataset._import_get_mask_location = lambda: original_get_mask_location


def _compute_garment_composite_mask(
    parsing_model,
    input_pil,
    result_pil,
    fallback_mask=None,
    feather_radius=15,
    dilation_size=5,
):
    """
    Soft alpha mask (L mode, 0-255; 255 = use generated result) computed by
    unioning the upper-garment parsing region from the input person image and
    the generated result, then feathering the boundary.  Falls back to
    fallback_mask (binary PIL image) if parsing finds no garment in either.
    """
    UPPER_GARMENT_LABEL = 4  # ATR label 4 = upper-clothes

    def _parse_garment(pil_img):
        small = pil_img.resize((384, 512), Image.LANCZOS)
        parsed, _ = parsing_model(small)
        arr = np.array(parsed)
        mask_arr = (arr == UPPER_GARMENT_LABEL).astype(np.uint8) * 255
        return Image.fromarray(mask_arr, mode='L').resize(pil_img.size, Image.NEAREST)

    mask_input = _parse_garment(input_pil)
    mask_result = _parse_garment(result_pil)
    union_arr = np.maximum(np.array(mask_input), np.array(mask_result))

    if union_arr.max() == 0 and fallback_mask is not None:
        union_arr = np.where(
            np.array(fallback_mask.convert('L')) > 127, 255, 0
        ).astype(np.uint8)

    mask_hard = Image.fromarray(union_arr, mode='L')
    mask = mask_hard
    if dilation_size > 0:
        mask = mask.filter(ImageFilter.MaxFilter(size=dilation_size))
    if feather_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))
        # Restore the original mask area to full opacity so the blur transition
        # only occurs in the buffer zone outside the original mask boundary.
        mask = Image.fromarray(
            np.maximum(np.array(mask), np.array(mask_hard)), mode='L'
        )
    return mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_densepose_predictor(device: str, model_path: str):
    import apply_net
    from detectron2.engine.defaults import DefaultPredictor
    args = apply_net.create_argument_parser().parse_args([
        "show",
        os.path.join(_REPO_ROOT, "configs", "densepose_rcnn_R_50_FPN_s1x.yaml"),
        model_path,
        "dp_segm", "-v",
        "--opts", "MODEL.DEVICE", device,
    ])
    cfg = apply_net.ShowAction.setup_config(args.cfg, args.model, args, [])
    predictor = DefaultPredictor(cfg)
    context = apply_net.ShowAction.create_context(args, cfg)
    return predictor, context


def _expand_conv_in_13ch(unet):
    """Expand unet.conv_in from 9 → 13 input channels (new channels zero-initialised)."""
    old = unet.conv_in
    conv_new = torch.nn.Conv2d(
        in_channels=13,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        padding=old.padding,
    )
    torch.nn.init.zeros_(conv_new.weight)
    conv_new.weight.data[:, :old.in_channels] = old.weight.data
    conv_new.bias.data = old.bias.data
    unet.conv_in = conv_new
    unet.config["in_channels"] = 13
    unet.config.in_channels = 13
    return unet


def _pad_to_multiple(t: torch.Tensor, multiple: int = 8):
    _, _, h, w = t.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return t, (0, 0)
    return torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="reflect"), (pad_h, pad_w)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="IDM-VTON Measurement Adapter inference on FIT dataset.")
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, default="yisol/IDM-VTON",
        help=(
            "Pretrained model. Must be the same base model used during training. "
            "Training uses yisol/IDM-VTON by default. "
            "The trained conv_in is an offset on top of these weights."
        ),
    )
    parser.add_argument(
        "--pretrained_garmentnet_path", type=str, default=None,
        help=(
            "GarmentNet base. Defaults to --pretrained_model_name_or_path with "
            "subfolder 'unet_encoder' (correct for yisol/IDM-VTON)."
        ),
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default=None,
        help=(
            "Path to a training checkpoint directory containing unet_lora.pt, "
            "measurement_adapter.pt, measurement_encoder.pt and adapter_config.json. "
            "If omitted the pretrained model is used as-is (no measurement conditioning)."
        ),
    )
    parser.add_argument(
        "--ip_adapter_path", type=str, default="ckpt/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin",
        help="Path to the IP-Adapter weights. Only needed when --checkpoint_dir is set.",
    )
    parser.add_argument(
        "--image_encoder_path", type=str, default=None,
        help="Path to the CLIP image encoder. Only needed when --checkpoint_dir is set.",
    )
    parser.add_argument(
        "--measurement_scale", type=float, default=1.0,
        help="Multiplier on the trained measurement-attention gates (0 = branch off).",
    )
    parser.add_argument(
        "--original_mask", action="store_true",
        help=(
            "Use the original IDM-VTON agnostic mask (no bust/length expansion from the "
            "measurements). Intended for the baseline run without --checkpoint_dir."
        ),
    )
    parser.add_argument("--data_dir", type=str, default="data/data_mini_test")
    parser.add_argument("--output_dir", type=str, default="result_adapter")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of dataset samples to process. -1 = all.")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=682)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_tokens", type=int, default=16,
                        help="Number of IP-Adapter image tokens (must match training).")
    parser.add_argument("--mixed_precision", type=str, default=None,
                        choices=["no", "fp16", "bf16"])
    parser.add_argument(
        "--densepose_path", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt", "densepose", "model_final_162be9.pkl"),
        help="Path to the DensePose checkpoint. Defaults to ckpt/densepose/ next to this script.",
    )
    parser.add_argument(
        "--parsing_gpu_id", type=int, default=0,
        help="GPU id passed to Parsing and OpenPose constructors.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device

    if args.seed is not None:
        set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    use_checkpoint = args.checkpoint_dir is not None

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    adapter_ctx = None
    measurement_encoder = None
    measurement_adapter = None

    if not use_checkpoint:
        # Pretrained IDM-VTON: no measurement conditioning.
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet",
            torch_dtype=weight_dtype,
        )
        unet_encoder = UNet2DConditionModel_ref.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet_encoder",
            torch_dtype=weight_dtype,
        )
        unet_encoder.config.addition_embed_type = None
        unet_encoder.config["addition_embed_type"] = None

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="image_encoder",
            torch_dtype=weight_dtype,
        )

        pipe = TryonPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet,
            unet_encoder=unet_encoder,
            image_encoder=image_encoder,
            feature_extractor=CLIPImageProcessor(),
            torch_dtype=weight_dtype,
            add_watermarker=False,
            safety_checker=None,
        ).to(device)

    else:
        # ── Checkpoint mode ───────────────────────────────────────────────────
        garmentnet_path = args.pretrained_garmentnet_path or args.pretrained_model_name_or_path
        garmentnet_subfolder = "unet" if args.pretrained_garmentnet_path else "unet_encoder"

        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False,
        )
        tokenizer_2 = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer_2", use_fast=False,
        )
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder",
        ).to(device, dtype=weight_dtype)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder_2",
        ).to(device, dtype=weight_dtype)
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae",
            torch_dtype=weight_dtype,
        ).to(device)

        if args.image_encoder_path:
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                args.image_encoder_path,
            ).to(device, dtype=weight_dtype)
        else:
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="image_encoder",
            ).to(device, dtype=weight_dtype)

        unet_encoder = UNet2DConditionModel_ref.from_pretrained(
            garmentnet_path, subfolder=garmentnet_subfolder,
        ).to(device, dtype=weight_dtype)
        unet_encoder.config.addition_embed_type = None
        unet_encoder.config["addition_embed_type"] = None
        unet_encoder.requires_grad_(False)

        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet",
            low_cpu_mem_usage=False, device_map=None,
        )

        # Load IP-Adapter weights
        ip_bin = args.ip_adapter_path or os.path.join(
            args.pretrained_model_name_or_path, "ip_adapter", "ip-adapter-plus_sdxl_vit-h.bin"
        )
        state_dict = torch.load(ip_bin, map_location="cpu")
        adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
        adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        image_proj_model = Resampler(
            dim=image_encoder.config.hidden_size,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=args.num_tokens,
            embedding_dim=image_encoder.config.hidden_size,
            output_dim=unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(device, dtype=torch.float32)
        image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        unet.encoder_hid_proj = image_proj_model

        if unet.conv_in.in_channels == 9:
            unet = _expand_conv_in_13ch(unet)

        with open(os.path.join(args.checkpoint_dir, "adapter_config.json")) as f:
            cfg = json.load(f)

        # conv_in (trained) from the checkpoint; the measurement modules are built afterwards.
        unet_state = torch.load(os.path.join(args.checkpoint_dir, "unet_lora.pt"), map_location="cpu")
        unet.load_state_dict(unet_state, strict=False)
        unet = unet.to(device, dtype=weight_dtype)
        unet.requires_grad_(False)

        measurement_encoder = MeasurementEncoder(
            num_measurements=9,
            hidden_dim=cfg["encoder_hidden_dim"],
            output_dim=cfg["encoder_output_dim"],
            dropout=0.1,
            use_fourier=cfg["use_fourier"],
            fourier_scale=cfg["fourier_scale"],
        )
        # Restores the Fourier projection buffer as well, so features match training.
        measurement_encoder.load_state_dict(
            torch.load(os.path.join(args.checkpoint_dir, "measurement_encoder.pt"), map_location="cpu")
        )
        measurement_encoder = measurement_encoder.to(device).eval()

        measurement_adapter = MeasurementAdapter(
            input_dim=cfg["encoder_output_dim"],
            hidden_sizes=get_hidden_sizes(unet),
            num_tokens=cfg["num_measurement_tokens"],
            token_dim=cfg["measurement_token_dim"],
            scale_mode=cfg["scale_mode"],
            num_timestep_bins=cfg["num_timestep_bins"],
        )
        measurement_adapter.load_state_dict(
            torch.load(os.path.join(args.checkpoint_dir, "measurement_adapter.pt"), map_location="cpu")
        )
        measurement_adapter.gate_table.data.mul_(args.measurement_scale)
        measurement_adapter = measurement_adapter.to(device, dtype=weight_dtype).eval()

        adapter_ctx = MeasurementAdapterContext()
        install_measurement_adapter(unet, measurement_adapter, adapter_ctx)

        pipe = TryonPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet,
            unet_encoder=unet_encoder,
            vae=vae,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            image_encoder=image_encoder,
            feature_extractor=CLIPImageProcessor(),
            torch_dtype=weight_dtype,
            add_watermarker=False,
            safety_checker=None,
        ).to(device)

    pipe.set_progress_bar_config(disable=False)

    def _align(v: int, m: int = 8) -> int:
        return v + (m - v % m) % m

    pad_h = _align(args.height) - args.height
    pad_w = _align(args.width) - args.width
    pipe_height = args.height + pad_h
    pipe_width = args.width + pad_w

    # ── Preprocessing models ──────────────────────────────────────────────────
    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose import OpenPose
    import apply_net
    from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation

    parsing_model = Parsing(args.parsing_gpu_id)
    openpose_model = OpenPose(args.parsing_gpu_id)
    openpose_model.preprocessor.body_estimation.model.to(device)

    densepose_predictor, densepose_context = _build_densepose_predictor(str(device), args.densepose_path)

    def densepose_fn(pil_image):
        img_bgr = convert_PIL_to_numpy(_apply_exif_orientation(pil_image), format="BGR")
        with torch.no_grad():
            instances = densepose_predictor(img_bgr)["instances"]
            vis = apply_net.ShowAction.execute_on_outputs(
                densepose_context, {"image": img_bgr}, instances
            )
        from PIL import Image as _Image
        return _Image.fromarray(vis[:, :, ::-1])

    # ── Dataset ───────────────────────────────────────────────────────────────
    if args.original_mask:
        _use_original_idm_vton_mask()
    dataset = FITDatasetWithMeasurements(
        data_root=args.data_dir,
        phase="inference",
        size=(args.height, args.width),
        parsing_model=parsing_model,
        openpose_model=openpose_model,
        densepose_fn=densepose_fn,
    )
    if args.num_samples > 0:
        indices = list(range(min(args.num_samples, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

    # ── Inference loop ────────────────────────────────────────────────────────
    for batch in dataloader:
        person_images = batch["person_image"].to(device, dtype=weight_dtype)
        garment_images = batch["garment_image"].to(device, dtype=weight_dtype)
        garment_clips = batch["garment_image_clip"]
        pose_imgs = batch["pose"].to(device, dtype=weight_dtype)
        masks = batch["mask"].to(device, dtype=weight_dtype)
        prompts = batch["text_prompts"]
        prompts_cloth = batch["text_prompts_cloth"]
        person_filenames = batch["person_filename"]

        bsz = person_images.shape[0]

        img_emb_list = [garment_clips[i] for i in range(bsz)]
        image_embeds = torch.cat(img_emb_list, dim=0).to(device, dtype=weight_dtype)

        if not isinstance(prompts, list):
            prompts = list(prompts)
        if not isinstance(prompts_cloth, list):
            prompts_cloth = list(prompts_cloth)
        negative_prompts = [negative_prompt] * bsz

        with torch.no_grad():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompts,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompts,
            )
            (prompt_embeds_c, _, _, _) = pipe.encode_prompt(
                prompts_cloth,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=negative_prompts,
            )

            # Measurement tokens -> side channel; the UNet pre-hook repeats them for CFG
            # and computes the timestep-dependent gates.
            if adapter_ctx is not None:
                measurements = batch["measurements"].to(device, dtype=torch.float32)
                emb = measurement_encoder(measurements, measurement_dropout_prob=0.0).squeeze(1)
                adapter_ctx.tokens = measurement_adapter(emb.to(dtype=weight_dtype))

            generator = torch.Generator(device).manual_seed(args.seed) if args.seed else None

            person_input = (person_images + 1.0) / 2.0
            if pad_h > 0 or pad_w > 0:
                person_input, _ = _pad_to_multiple(person_input)
                pose_imgs_in, _ = _pad_to_multiple(pose_imgs)
                masks_in, _ = _pad_to_multiple(masks)
                garment_images_in, _ = _pad_to_multiple(garment_images)
            else:
                pose_imgs_in = pose_imgs
                masks_in = masks
                garment_images_in = garment_images

            images = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                strength=1.0,
                pose_img=pose_imgs_in,
                text_embeds_cloth=prompt_embeds_c,
                cloth=garment_images_in,
                mask_image=masks_in,
                image=person_input,
                height=pipe_height,
                width=pipe_width,
                guidance_scale=args.guidance_scale,
                ip_adapter_image=image_embeds,
            )[0]

            if adapter_ctx is not None:
                adapter_ctx.tokens = None

            if pad_h > 0 or pad_w > 0:
                images = [img.crop((0, 0, args.width, args.height)) for img in images]

        crop_boxes = batch["person_crop_box"]
        input_person_filenames = batch["input_person_filename"]

        for i, (img, fname) in enumerate(zip(images, person_filenames)):
            stem = os.path.splitext(os.path.basename(fname))[0]
            out_path = os.path.join(args.output_dir, f"{stem}_result.png")

            left, top, right, bottom = (int(crop_boxes[k][i]) for k in range(4))
            crop_w = right - left
            crop_h = bottom - top

            orig_person = Image.open(
                os.path.join(args.data_dir, input_person_filenames[i])
            ).convert("RGB")
            result_resized = img.resize((crop_w, crop_h), Image.LANCZOS)

            res_path = os.path.join(args.output_dir, f"{stem}_generated.png")
            result_resized.save(res_path)
            print(f"Saved: {res_path}")

            # Upscale the agnostic mask to crop size (used as fallback if parsing fails).
            mask_crop = Image.fromarray(
                masks[i, 0].cpu().float().mul(255).byte().numpy()
            ).resize((crop_w, crop_h), Image.NEAREST)

            orig_crop = orig_person.crop((left, top, right, bottom))

            # Build a tight compositing mask: union of garment regions parsed from
            # the input person and the generated result, then feathered at the edges.
            composite_mask = _compute_garment_composite_mask(
                parsing_model, orig_crop, result_resized, fallback_mask=mask_crop,
                feather_radius=15,
            )

            # Alpha-blend: generated where mask=255, original where mask=0.
            result_composed = Image.composite(result_resized, orig_crop, composite_mask)
            orig_person.paste(result_composed, (left, top))
            orig_person.save(out_path)
            print(f"Saved: {out_path}")

            # Save inpainting mask as a semi-transparent red overlay on the input crop.
            mask_out = os.path.join(args.output_dir, f"{stem}_mask.png")
            inpaint_mask_pil = Image.fromarray(
                masks[i, 0].cpu().float().mul(255).byte().numpy()
            ).resize((crop_w, crop_h), Image.NEAREST)
            red_layer = Image.new("RGB", orig_crop.size, (220, 50, 50))
            alpha = inpaint_mask_pil.point(lambda p: int(p * 0.45))
            mask_viz = Image.composite(red_layer, orig_crop, alpha)
            mask_viz.save(mask_out)
            print(f"Saved: {mask_out}")


if __name__ == "__main__":
    main()
