"""
inference_fit_facerestore.py — IDM-VTON inference with face+hair restoration.

Compositing strategy: use the full generated image as the base (so the garment
looks exactly as generated with no seams at the garment boundary), then paste
back only the face and hair regions from the input person image.

This avoids the core problem of all garment-boundary compositing approaches
(seams at cuffs, arm position mismatches, agnostic mask overreach) by moving
the compositing boundary far away from the garment — into the hair edges and
forehead — where a small feather blends naturally.

The face+hair mask is produced by the ATR humanparsing model that is already
loaded for preprocessing (labels 2=hair, 11=face), so no new models are needed.

Two modes:

  Without --checkpoint_dir (default):
    Loads yisol/IDM-VTON pretrained weights and runs inference on FIT data.
    Measurements are read but not injected (the pretrained model has no
    measurement conditioning). Use this to verify the data pipeline and get a
    baseline before any fine-tuning.

  With --checkpoint_dir <path>:
    Loads the base model specified by --pretrained_model_name_or_path (must be
    the SAME base used during training, default yisol/IDM-VTON), then overlays
    LoRA + conv_in + MeasurementEncoder weights from the checkpoint. Measurement
    tokens are appended to the cross-attention sequence
    ([B, 77, 2048] → [B, 78, 2048]) for positive prompt; zero token for negative/CFG.

    IMPORTANT: --pretrained_model_name_or_path must match the training base model.
    Training uses yisol/IDM-VTON (13-channel UNet). Using a different base
    (e.g. diffusers/stable-diffusion-xl-1.0-inpainting-0.1) will corrupt outputs
    because LoRA deltas are additive offsets on top of the training base weights.

Example (quick single-sample test, no checkpoint):
    python inference_fit_facerestore.py --data_dir data/data_mini_test --num_samples 1

Example (after training):
    python inference_fit_facerestore.py \\
        --data_dir data/data_mini_test \\
        --checkpoint_dir output/checkpoint-500 \\
        --output_dir result_fit_facerestore \\
        --num_samples 4
"""

import argparse
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
from src.measurement_encoder import MeasurementEncoder
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="IDM-VTON inference with face+hair restoration compositing."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, default="yisol/IDM-VTON",
        help=(
            "Pretrained model. Must be the same base model used during training. "
            "Training uses yisol/IDM-VTON by default, so inference should too. "
            "The LoRA deltas are additive offsets on top of these weights — using a "
            "different base (e.g. diffusers/stable-diffusion-xl-1.0-inpainting-0.1) "
            "will produce corrupted results."
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
            "Path to a training checkpoint directory containing lora/, conv_in.pt, "
            "and measurement_encoder.pt. If omitted the pretrained model is used as-is."
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
    parser.add_argument("--data_dir", type=str, default="data/data_mini_test")
    parser.add_argument("--output_dir", type=str, default="result_fit_facerestore")
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
    parser.add_argument(
        "--face_feather_radius", type=int, default=8,
        help="Gaussian blur radius for feathering the face+hair mask boundary (default 8).",
    )
    return parser.parse_args()


def _pad_to_multiple(t: torch.Tensor, multiple: int = 8):
    """Pad a [B, C, H, W] tensor so H and W are multiples of `multiple`.
    Returns (padded_tensor, (pad_bottom, pad_right)) so the crop can be undone."""
    _, _, h, w = t.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return t, (0, 0)
    return torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="reflect"), (pad_h, pad_w)


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


def _face_restore_mask(parsing_model, input_pil, feather_radius=8):
    """
    Soft alpha mask (L mode, 0-255; 255 = use input person) covering the face
    and hair regions parsed from input_pil, feathered at the boundary.

    Uses ATR labels 2 (hair) and 11 (face).  Neck is intentionally excluded
    because its boundary sits at the garment neckline where seams would show.
    The feathered boundary falls in the hair edges and forehead — far from the
    garment — where blending is naturally smooth.
    """
    FACE_HAIR_LABELS = [2, 11]  # ATR: 2=hair, 11=face

    small = input_pil.resize((384, 512), Image.LANCZOS)
    parsed, _ = parsing_model(small)
    arr = np.array(parsed)
    mask_arr = np.isin(arr, FACE_HAIR_LABELS).astype(np.uint8) * 255
    mask_hard = Image.fromarray(mask_arr, mode='L').resize(input_pil.size, Image.NEAREST)

    if feather_radius <= 0:
        return mask_hard

    mask = mask_hard.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    # Keep the interior of the mask at full opacity so the blur only affects
    # the boundary — the same buffer trick used elsewhere in this project.
    mask = Image.fromarray(
        np.maximum(np.array(mask), np.array(mask_hard)), mode='L'
    )
    return mask


def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device

    if args.seed is not None:
        set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    use_checkpoint = args.checkpoint_dir is not None

    # ── Load models ───────────────────────────────────────────────────────────
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if not use_checkpoint:
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
        measurement_encoder = None

    else:
        # ── Checkpoint mode ───────────────────────────────────────────────────
        from peft import PeftModel

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

        unet = PeftModel.from_pretrained(unet, os.path.join(args.checkpoint_dir, "lora"))
        unet = unet.merge_and_unload()

        conv_state = torch.load(
            os.path.join(args.checkpoint_dir, "conv_in.pt"), map_location="cpu"
        )
        unet.conv_in.load_state_dict(conv_state)

        unet = unet.to(device, dtype=weight_dtype)
        unet.requires_grad_(False)

        cross_attn_dim = unet.config.cross_attention_dim
        measurement_encoder = MeasurementEncoder(
            num_measurements=9,
            hidden_dim=256,
            output_dim=cross_attn_dim,
            dropout=0.1,
        ).to(device)
        menc_state = torch.load(
            os.path.join(args.checkpoint_dir, "measurement_encoder.pt"), map_location="cpu"
        )
        measurement_encoder.load_state_dict(menc_state)
        measurement_encoder.eval()

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

    # ── Derive padded dimensions (VAE requires multiples of 8) ───────────────
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

            if measurement_encoder is not None:
                measurements = batch["measurements"].to(device, dtype=torch.float32)
                measurement_tokens = measurement_encoder(measurements)
                measurement_tokens = measurement_tokens.to(dtype=prompt_embeds.dtype)
                null_tokens = torch.zeros_like(measurement_tokens)

                prompt_embeds = torch.cat([prompt_embeds, measurement_tokens], dim=1)
                negative_prompt_embeds = torch.cat([negative_prompt_embeds, null_tokens], dim=1)

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

            orig_crop = orig_person.crop((left, top, right, bottom))

            # Build face+hair mask from the input person, then paste those
            # regions back onto the generated image.  The garment region is
            # taken entirely from the generated image so there are no seams at
            # the garment boundary.
            face_mask = _face_restore_mask(
                parsing_model, orig_crop,
                feather_radius=args.face_feather_radius,
            )

            # image1=orig_crop where mask=255 (face+hair), image2=result_resized elsewhere.
            result_composed = Image.composite(orig_crop, result_resized, face_mask)
            orig_person.paste(result_composed, (left, top))
            orig_person.save(out_path)
            print(f"Saved: {out_path}")

            # Save face+hair mask as a semi-transparent red overlay on the input crop.
            mask_out = os.path.join(args.output_dir, f"{stem}_mask.png")
            red_layer = Image.new("RGB", orig_crop.size, (220, 50, 50))
            alpha = face_mask.point(lambda p: int(p * 0.45))
            mask_viz = Image.composite(red_layer, orig_crop, alpha)
            mask_viz.save(mask_out)
            print(f"Saved: {mask_out}")


if __name__ == "__main__":
    main()
