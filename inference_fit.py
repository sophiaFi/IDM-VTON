"""
inference_fit.py — Run IDM-VTON inference on FIT dataset samples.

Two modes:

  Without --checkpoint_dir (default):
    Loads yisol/IDM-VTON pretrained weights and runs inference on FIT data.
    Measurements are read but not injected (the pretrained model has no
    measurement conditioning). Use this to verify the data pipeline and get a
    baseline before any fine-tuning.

  With --checkpoint_dir <path>:
    Loads the base model specified by --pretrained_model_name_or_path, expands
    conv_in to 13 channels, then overlays LoRA + conv_in + MeasurementEncoder
    weights from the checkpoint. Measurement tokens are appended to the
    cross-attention sequence ([B, 77, 2048] → [B, 78, 2048]) for both the
    positive and negative prompt paths (zero token for negative/CFG).

Example (quick single-sample test, no checkpoint):
    python inference_fit.py --data_dir data_mini_test --num_samples 1

Example (after training):
    python inference_fit.py \
        --data_dir data_mini_test \
        --checkpoint_dir output/checkpoint-500 \
        --pretrained_model_name_or_path diffusers/stable-diffusion-xl-1.0-inpainting-0.1 \
        --pretrained_garmentnet_path stabilityai/stable-diffusion-xl-base-1.0 \
        --output_dir result_fit \
        --num_samples 4
"""

import argparse
import os
import sys
from typing import List

import torch
import torchvision
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

from ip_adapter.ip_adapter import Resampler
from src.fit_dataset import FITDatasetWithMeasurements
from src.measurement_encoder import MeasurementEncoder
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel


def parse_args():
    parser = argparse.ArgumentParser(description="IDM-VTON inference on FIT dataset.")
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, default="yisol/IDM-VTON",
        help=(
            "Pretrained model. Default (yisol/IDM-VTON) is used when running without a "
            "checkpoint. When --checkpoint_dir is set, use the same base model that was "
            "used during training (e.g. diffusers/stable-diffusion-xl-1.0-inpainting-0.1)."
        ),
    )
    parser.add_argument(
        "--pretrained_garmentnet_path", type=str, default=None,
        help=(
            "GarmentNet base. Defaults to --pretrained_model_name_or_path with "
            "subfolder 'unet_encoder' (works for yisol/IDM-VTON). When using a raw "
            "SDXL base with a checkpoint, set this to "
            "stabilityai/stable-diffusion-xl-base-1.0."
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
        "--image_encoder_path", type=str, default="ckpt/image_encoder",
        help="Path to the CLIP image encoder. Only needed when --checkpoint_dir is set.",
    )
    parser.add_argument("--data_dir", type=str, default="data_mini_test")
    parser.add_argument("--output_dir", type=str, default="result_fit")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of dataset samples to process. -1 = all.")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_tokens", type=int, default=16,
                        help="Number of IP-Adapter image tokens (must match training).")
    parser.add_argument("--mixed_precision", type=str, default=None,
                        choices=["no", "fp16", "bf16"])
    return parser.parse_args()


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
        # Pretrained IDM-VTON: all submodels (including image_encoder and
        # unet_encoder) live inside yisol/IDM-VTON. Load them explicitly so the
        # pipeline doesn't try to resolve them from config files that don't exist.
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

        garmentnet_path = args.pretrained_garmentnet_path or "stabilityai/stable-diffusion-xl-base-1.0"

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
            torch_dtype=torch.float16,
        ).to(device)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            args.image_encoder_path,
        ).to(device, dtype=weight_dtype)

        unet_encoder = UNet2DConditionModel_ref.from_pretrained(
            garmentnet_path, subfolder="unet",
        ).to(device, dtype=weight_dtype)
        unet_encoder.config.addition_embed_type = None
        unet_encoder.config["addition_embed_type"] = None
        unet_encoder.requires_grad_(False)

        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet",
            low_cpu_mem_usage=False, device_map=None,
        )
        unet.config.encoder_hid_dim = image_encoder.config.hidden_size
        unet.config.encoder_hid_dim_type = "ip_image_proj"
        unet.config["encoder_hid_dim"] = image_encoder.config.hidden_size
        unet.config["encoder_hid_dim_type"] = "ip_image_proj"

        # IP-Adapter weights
        state_dict = torch.load(args.ip_adapter_path, map_location="cpu")
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

        # Expand conv_in 9 → 13
        unet = _expand_conv_in_13ch(unet)

        # Load LoRA adapter
        unet = PeftModel.from_pretrained(unet, os.path.join(args.checkpoint_dir, "lora"))
        unet = unet.merge_and_unload()

        # Load conv_in weights
        conv_state = torch.load(
            os.path.join(args.checkpoint_dir, "conv_in.pt"), map_location="cpu"
        )
        unet.conv_in.load_state_dict(conv_state)

        unet = unet.to(device, dtype=weight_dtype)
        unet.requires_grad_(False)

        # Load MeasurementEncoder
        cross_attn_dim = unet.config.cross_attention_dim
        measurement_encoder = MeasurementEncoder(
            num_measurements=7,
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

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = FITDatasetWithMeasurements(
        data_root=args.data_dir,
        phase="test",
        size=(args.height, args.width),
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
            # Encode text prompts
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

            # Inject measurement tokens when a checkpoint is loaded
            if measurement_encoder is not None:
                measurements = batch["measurements"].to(device, dtype=torch.float32)
                measurement_tokens = measurement_encoder(measurements)  # [B, 1, cross_attn_dim]
                measurement_tokens = measurement_tokens.to(dtype=prompt_embeds.dtype)
                null_tokens = torch.zeros_like(measurement_tokens)

                prompt_embeds = torch.cat([prompt_embeds, measurement_tokens], dim=1)
                negative_prompt_embeds = torch.cat([negative_prompt_embeds, null_tokens], dim=1)

            generator = torch.Generator(device).manual_seed(args.seed) if args.seed else None

            images = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                strength=1.0,
                pose_img=pose_imgs,
                text_embeds_cloth=prompt_embeds_c,
                cloth=garment_images,
                mask_image=masks,
                image=(person_images + 1.0) / 2.0,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                ip_adapter_image=image_embeds,
            )[0]

        for i, (img, fname) in enumerate(zip(images, person_filenames)):
            stem = os.path.splitext(os.path.basename(fname))[0]
            out_path = os.path.join(args.output_dir, f"{stem}_result.png")
            img.save(out_path)
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
