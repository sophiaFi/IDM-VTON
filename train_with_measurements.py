"""
Training script for IDM-VTON + Measurement Encoder on the FIT dataset.

Extends train_xl.py by:
- Using FITDatasetWithMeasurements (src/fit_dataset.py) as the training dataset.
- Instantiating MeasurementEncoder (src/measurement_encoder.py) with output_dim=2048
  to match SDXL's cross-attention dim (CLIP-L 768 + OpenCLIP 1280).
- Concatenating the measurement token to the text cross-attention sequence before
  every UNet forward pass: encoder_hidden_states [B, 77, 2048] -> [B, 78, 2048].

Parameter-efficient fine-tuning strategy:
- LoRA (rank 4 by default) on all attention Q/K/V/out projections in TryonNet.
  to_k_ip / to_v_ip (IP-Adapter) are NOT targeted — they keep their pretrained weights.
- The expanded conv_in (9→13 channels, new densepose channels zero-initialized) is
  trained in full so the model can learn to use densepose from scratch.
- MeasurementEncoder is trained in full (new module, no pretrained weights).
- Everything else (VAE, text encoders, image encoder, GarmentNet, Resampler) is frozen.

Checkpoint layout per saved epoch:
  checkpoint-<step>/
    lora/                   ← LoRA adapter weights (peft format)
    conv_in.pt              ← trained conv_in state dict
    measurement_encoder.pt  ← MeasurementEncoder state dict
"""

import os
import json
import time
import itertools
import math
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import (
    CLIPTextModel, CLIPTokenizer,
    CLIPVisionModelWithProjection, CLIPTextModelWithProjection,
)
#from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
from typing import List

from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.fit_dataset import FITDatasetWithMeasurements
from src.measurement_encoder import MeasurementEncoder

from ip_adapter.ip_adapter import Resampler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.training_utils import compute_snr


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="IDM-VTON + LoRA + MeasurementEncoder training on FIT dataset.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    parser.add_argument("--pretrained_garmentnet_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--pretrained_ip_adapter_path", type=str, default="ckpt/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin")
    parser.add_argument("--image_encoder_path", type=str, default="ckpt/image_encoder")
    parser.add_argument("--data_dir", type=str, default="gs://ma-idm-vton-data/datasets/fit-mini")
    parser.add_argument("--output_dir", type=str, default="output")
    # LoRA
    parser.add_argument("--lora_rank", type=int, default=4,
                        help="LoRA rank for cross-attention projections.")
    parser.add_argument("--lora_alpha", type=int, default=4,
                        help="LoRA alpha (scaling = alpha / rank). Use alpha == rank for scale 1.")
    # Image dimensions
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1024)
    # Training
    parser.add_argument("--train_batch_size", type=int, default=6)
    parser.add_argument("--test_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=130)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Higher LR than full fine-tuning is typical for LoRA.")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--snr_gamma", type=float, default=None)
    parser.add_argument("--noise_offset", type=float, default=None)
    parser.add_argument("--measurement_dropout", type=float, default=0.1)
    # Checkpointing / logging
    parser.add_argument("--checkpointing_epoch", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=1000)
    # Efficiency
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    # IP-Adapter
    parser.add_argument("--num_tokens", type=int, default=16)
    # Inference (validation samples)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    # Misc
    parser.add_argument("--local_rank", type=int, default=-1)
    # Resource profiling: run for N steps then write resource_profile.json and exit.
    # Set to 0 (default) to disable and run a full training.
    parser.add_argument("--profile_steps", type=int, default=0,
                        help="If > 0, run this many steps, save a resource profile, and exit.")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


def main():
    args = parse_args()
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=ProjectConfiguration(project_dir=args.output_dir),
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────────
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", rescale_betas_zero_snr=True
    )
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=torch.float16)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)

    unet_encoder = UNet2DConditionModel_ref.from_pretrained(args.pretrained_garmentnet_path, subfolder="unet")
    unet_encoder.config.addition_embed_type = None
    unet_encoder.config["addition_embed_type"] = None

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", low_cpu_mem_usage=False, device_map=None
    )
    unet.config.encoder_hid_dim = image_encoder.config.hidden_size
    unet.config.encoder_hid_dim_type = "ip_image_proj"
    unet.config["encoder_hid_dim"] = image_encoder.config.hidden_size
    unet.config["encoder_hid_dim_type"] = "ip_image_proj"

    # ── IP-Adapter weights → attention processors ─────────────────────────────
    state_dict = torch.load(args.pretrained_ip_adapter_path, map_location="cpu")
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

    # Resampler: projects CLIP image features → IP tokens [B, num_tokens, cross_attn_dim]
    # Frozen — the garment appearance path is already well-trained in IDM-VTON.
    image_proj_model = Resampler(
        dim=image_encoder.config.hidden_size,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=args.num_tokens,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4,
    ).to(accelerator.device, dtype=torch.float32)
    image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
    unet.encoder_hid_proj = image_proj_model

    # ── Expand TryonNet input conv 9 → 13 channels ───────────────────────────
    # Channels: noisy latent (4) | mask (1) | masked-image latent (4) | densepose latent (4)
    # New columns (9-12) are zero-initialized so the model starts ignoring densepose
    # and gradually learns to use it.
    conv_new = torch.nn.Conv2d(
        in_channels=13, out_channels=unet.conv_in.out_channels, kernel_size=3, padding=1
    )
    torch.nn.init.kaiming_normal_(conv_new.weight)
    conv_new.weight.data *= 0.0
    conv_new.weight.data[:, :9] = unet.conv_in.weight.data
    conv_new.bias.data = unet.conv_in.bias.data
    unet.conv_in = conv_new
    unet.config["in_channels"] = 13
    unet.config.in_channels = 13

    # ── MeasurementEncoder ────────────────────────────────────────────────────
    cross_attn_dim = unet.config.cross_attention_dim  # 2048 for SDXL
    measurement_encoder = MeasurementEncoder(
        num_measurements=7,
        hidden_dim=256,
        output_dim=cross_attn_dim,
        dropout=0.1,
        use_fourier=False,
    )

    # ── dtype / device placement ──────────────────────────────────────────────
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    unet_encoder.to(accelerator.device, dtype=weight_dtype)

    # ── Memory optimizations (must happen before LoRA wrapping) ───────────────
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly.")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        unet_encoder.enable_gradient_checkpointing()

    # ── Freeze everything, then apply LoRA selectively ───────────────────────
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet_encoder.requires_grad_(False)
    image_proj_model.requires_grad_(False)
    unet.requires_grad_(False)

    # LoRA on all attention Q/K/V/out projections (both self- and cross-attention).
    # "to_k" / "to_v" do NOT match "to_k_ip" / "to_v_ip" (suffix mismatch), so
    # the IP-Adapter key/value projections remain frozen.
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)

    # conv_in must be trainable separately: LoRA does not cover Conv2d layers and
    # the 4 new densepose channels need to be learned from zero.
    unet.base_model.model.conv_in.requires_grad_(True)

    measurement_encoder.requires_grad_(True)

    # ── Optimizer (LoRA params + conv_in + MeasurementEncoder) ───────────────
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Install bitsandbytes to use 8-bit Adam: `pip install bitsandbytes`.")
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    trainable_unet_params = [p for p in unet.parameters() if p.requires_grad]
    params_to_opt = itertools.chain(trainable_unet_params, measurement_encoder.parameters())
    optimizer = optimizer_class(
        params_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if accelerator.is_main_process:
        n_lora = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        n_menc = sum(p.numel() for p in measurement_encoder.parameters())
        print(f"Trainable params — LoRA+conv_in: {n_lora:,}  MeasurementEncoder: {n_menc:,}  total: {n_lora + n_menc:,}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = FITDatasetWithMeasurements(
        data_root=args.data_dir, phase="train", size=(args.height, args.width)
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, pin_memory=True, shuffle=True,
        batch_size=args.train_batch_size, num_workers=4,
    )
    test_dataset = FITDatasetWithMeasurements(
        data_root=args.data_dir, phase="test", size=(args.height, args.width)
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, batch_size=args.test_batch_size, num_workers=4,
    )

    # ── Step count ────────────────────────────────────────────────────────────
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    overrode_max_train_steps = args.max_train_steps is None
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # ── Accelerator prepare ───────────────────────────────────────────────────
    (
        unet,
        measurement_encoder,
        image_proj_model,
        unet_encoder,
        image_encoder,
        optimizer,
        train_dataloader,
        test_dataloader,
    ) = accelerator.prepare(
        unet, measurement_encoder, image_proj_model,
        unet_encoder, image_encoder,
        optimizer, train_dataloader, test_dataloader,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # ── Training loop ─────────────────────────────────────────────────────────
    profiling = args.profile_steps > 0
    profile_step_times = []

    if profiling and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    max_steps = args.profile_steps if profiling else args.max_train_steps
    progress_bar = tqdm(
        range(max_steps), desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    global_step = 0
    train_loss = 0.0

    unet.train()
    measurement_encoder.train()

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            step_start = time.perf_counter()
            with accelerator.accumulate(unet), accelerator.accumulate(measurement_encoder):

                # ── Validation samples ────────────────────────────────────────
                if global_step % args.logging_steps == 0 and accelerator.is_main_process:
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        # merge_and_unload() returns a plain UNet2DConditionModel with
                        # LoRA weights merged in, leaving the PeftModel unet unchanged.
                        merged_unet = accelerator.unwrap_model(unet).merge_and_unload()
                        newpipe = TryonPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=merged_unet,
                            vae=vae,
                            scheduler=noise_scheduler,
                            tokenizer=tokenizer,
                            tokenizer_2=tokenizer_2,
                            text_encoder=text_encoder,
                            text_encoder_2=text_encoder_2,
                            image_encoder=image_encoder,
                            unet_encoder=unet_encoder,
                            torch_dtype=torch.float16,
                            add_watermarker=False,
                            safety_checker=None,
                        ).to(accelerator.device)

                        for sample in test_dataloader:
                            img_emb_list = [sample["garment_image_clip"][i] for i in range(sample["garment_image_clip"].shape[0])]
                            num_prompts = len(img_emb_list)
                            prompt = sample["text_prompts"]
                            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                            if not isinstance(prompt, List):
                                prompt = [prompt] * num_prompts
                            if not isinstance(negative_prompt, List):
                                negative_prompt = [negative_prompt] * num_prompts

                            image_embeds = torch.cat(img_emb_list, dim=0)

                            with torch.inference_mode():
                                (
                                    prompt_embeds, negative_prompt_embeds,
                                    pooled_prompt_embeds, negative_pooled_prompt_embeds,
                                ) = newpipe.encode_prompt(
                                    prompt, num_images_per_prompt=1,
                                    do_classifier_free_guidance=True,
                                    negative_prompt=negative_prompt,
                                )
                                prompt_cloth = sample["text_prompts_cloth"]
                                negative_prompt_cloth = "monochrome, lowres, bad anatomy, worst quality, low quality"
                                if not isinstance(prompt_cloth, List):
                                    prompt_cloth = [prompt_cloth] * num_prompts
                                if not isinstance(negative_prompt_cloth, List):
                                    negative_prompt_cloth = [negative_prompt_cloth] * num_prompts
                                (prompt_embeds_c, _, _, _) = newpipe.encode_prompt(
                                    prompt_cloth, num_images_per_prompt=1,
                                    do_classifier_free_guidance=False,
                                    negative_prompt=negative_prompt_cloth,
                                )
                                generator = torch.Generator(newpipe.device).manual_seed(args.seed) if args.seed else None
                                images = newpipe(
                                    prompt_embeds=prompt_embeds,
                                    negative_prompt_embeds=negative_prompt_embeds,
                                    pooled_prompt_embeds=pooled_prompt_embeds,
                                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                                    num_inference_steps=args.num_inference_steps,
                                    generator=generator,
                                    strength=1.0,
                                    pose_img=sample["pose"],
                                    text_embeds_cloth=prompt_embeds_c,
                                    cloth=sample["garment_image"].to(accelerator.device),
                                    mask_image=sample["mask"],
                                    image=(sample["person_image"] + 1.0) / 2.0,
                                    height=args.height,
                                    width=args.width,
                                    guidance_scale=args.guidance_scale,
                                    ip_adapter_image=image_embeds,
                                )[0]
                            for i, img in enumerate(images):
                                img.save(os.path.join(args.output_dir, f"{global_step}_{i}_test.jpg"))
                            break

                        del merged_unet, newpipe
                        torch.cuda.empty_cache()

                # ── Encode images to latents ──────────────────────────────────
                pixel_values = batch["person_image"].to(dtype=vae.dtype)
                model_input = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

                masked_latents = vae.encode(
                    batch["masked_person"].reshape(batch["person_image"].shape).to(dtype=vae.dtype)
                ).latent_dist.sample() * vae.config.scaling_factor

                masks = batch["mask"]
                mask = torch.nn.functional.interpolate(
                    masks, size=(args.height // 8, args.width // 8)
                ).reshape(-1, 1, args.height // 8, args.width // 8)

                pose_map = vae.encode(batch["pose"].to(dtype=vae.dtype)).latent_dist.sample() * vae.config.scaling_factor

                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                )
                noisy_latents = noise_scheduler.add_noise(model_input, noise, timesteps)
                latent_model_input = torch.cat([noisy_latents, mask, masked_latents, pose_map], dim=1)  # [B, 13, H/8, W/8]

                # ── Person text embeddings ────────────────────────────────────
                text_input_ids = tokenizer(
                    batch["text_prompts"], max_length=tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt",
                ).input_ids
                text_input_ids_2 = tokenizer_2(
                    batch["text_prompts"], max_length=tokenizer_2.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt",
                ).input_ids
                enc1 = text_encoder(text_input_ids.to(accelerator.device), output_hidden_states=True)
                enc2 = text_encoder_2(text_input_ids_2.to(accelerator.device), output_hidden_states=True)
                text_embeds = enc1.hidden_states[-2]           # [B, 77, 768]
                pooled_text_embeds = enc2[0]                   # [B, 1280]
                text_embeds_2 = enc2.hidden_states[-2]         # [B, 77, 1280]
                encoder_hidden_states = torch.cat([text_embeds, text_embeds_2], dim=-1)  # [B, 77, 2048]

                # ── Measurement token ─────────────────────────────────────────
                measurements = batch["measurements"].to(accelerator.device, dtype=torch.float32)
                measurement_tokens = measurement_encoder(
                    measurements, measurement_dropout_prob=args.measurement_dropout
                ).to(dtype=encoder_hidden_states.dtype)          # [B, 1, 2048]
                encoder_hidden_states = torch.cat([encoder_hidden_states, measurement_tokens], dim=1)  # [B, 78, 2048]

                # ── SDXL time/size conditioning ───────────────────────────────
                def compute_time_ids(original_size, crops_coords_top_left=(0, 0)):
                    add_time_ids = list(original_size + crops_coords_top_left + (args.height, args.height))
                    return torch.tensor([add_time_ids]).to(accelerator.device)

                add_time_ids = torch.cat([compute_time_ids((args.height, args.height)) for _ in range(bsz)])

                # ── IP-Adapter: garment CLIP image embeddings ─────────────────
                image_embeds = torch.cat([batch["garment_image_clip"][i] for i in range(bsz)], dim=0)
                image_embeds = image_encoder(image_embeds, output_hidden_states=True).hidden_states[-2]
                ip_tokens = image_proj_model(image_embeds)

                unet_added_cond_kwargs = {
                    "text_embeds": pooled_text_embeds,
                    "time_ids": add_time_ids,
                    "image_embeds": ip_tokens,
                }

                # ── GarmentNet → reference features ──────────────────────────
                cloth_values = vae.encode(
                    batch["garment_image"].to(accelerator.device, dtype=vae.dtype)
                ).latent_dist.sample() * vae.config.scaling_factor

                text_input_ids_c = tokenizer(
                    batch["text_prompts_cloth"], max_length=tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt",
                ).input_ids
                text_input_ids_2_c = tokenizer_2(
                    batch["text_prompts_cloth"], max_length=tokenizer_2.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt",
                ).input_ids
                enc1_c = text_encoder(text_input_ids_c.to(accelerator.device), output_hidden_states=True)
                enc2_c = text_encoder_2(text_input_ids_2_c.to(accelerator.device), output_hidden_states=True)
                text_embeds_cloth = torch.cat([enc1_c.hidden_states[-2], enc2_c.hidden_states[-2]], dim=-1)

                _, reference_features = unet_encoder(cloth_values, timesteps, text_embeds_cloth, return_dict=False)
                reference_features = list(reference_features)

                # ── TryonNet forward pass ─────────────────────────────────────
                noise_pred = unet(
                    latent_model_input, timesteps, encoder_hidden_states,
                    added_cond_kwargs=unet_added_cond_kwargs,
                    garment_features=reference_features,
                ).sample

                # ── Loss ──────────────────────────────────────────────────────
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                elif noise_scheduler.config.prediction_type == "sample":
                    target = model_input
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.snr_gamma is None:
                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
                else:
                    snr = compute_snr(noise_scheduler, timesteps)
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        snr = snr + 1
                    weights = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    loss = (F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                            .mean(dim=list(range(1, noise_pred.ndim))) * weights).mean()

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    trainable_params = (
                        [p for p in unet.parameters() if p.requires_grad]
                        + list(measurement_encoder.parameters())
                    )
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                global_step += 1
                if profiling:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    profile_step_times.append(time.perf_counter() - step_start)

            if accelerator.sync_gradients:
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

            progress_bar.set_postfix(step_loss=loss.detach().item())

            if global_step >= max_steps:
                break

        # ── Profile report (written once, then exit) ──────────────────────────
        if profiling and global_step >= max_steps and accelerator.is_main_process:
            report = {
                "profile_steps": args.profile_steps,
                "train_batch_size": args.train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_batch_size": args.train_batch_size * args.gradient_accumulation_steps,
                "mixed_precision": args.mixed_precision,
                "gradient_checkpointing": args.gradient_checkpointing,
                "enable_xformers": args.enable_xformers_memory_efficient_attention,
                "lora_rank": args.lora_rank,
            }
            if profile_step_times:
                times = profile_step_times[1:]  # drop first step (includes dataloader warm-up)
                if times:
                    avg_s = sum(times) / len(times)
                    report["step_time_avg_s"] = round(avg_s, 3)
                    report["step_time_min_s"] = round(min(times), 3)
                    report["step_time_max_s"] = round(max(times), 3)
                    report["throughput_samples_per_s"] = round(args.train_batch_size / avg_s, 2)
            if torch.cuda.is_available():
                peak_bytes = torch.cuda.max_memory_allocated()
                reserved_bytes = torch.cuda.max_memory_reserved()
                report["gpu_peak_allocated_gb"] = round(peak_bytes / 1024**3, 2)
                report["gpu_peak_reserved_gb"] = round(reserved_bytes / 1024**3, 2)
                props = torch.cuda.get_device_properties(0)
                report["gpu_name"] = props.name
                report["gpu_total_memory_gb"] = round(props.total_memory / 1024**3, 2)

            report_path = os.path.join(args.output_dir, "resource_profile.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nResource profile written to {report_path}")
            print(json.dumps(report, indent=2))
            break  # exit epoch loop after profiling

        # ── Checkpoint ────────────────────────────────────────────────────────
        if epoch % args.checkpointing_epoch == 0 and accelerator.is_main_process:
            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            os.makedirs(save_path, exist_ok=True)

            unwrapped_peft_unet = accelerator.unwrap_model(unet)

            # LoRA adapter weights (peft format — small, just the delta matrices)
            unwrapped_peft_unet.save_pretrained(os.path.join(save_path, "lora"))

            # conv_in weights (trained Conv2d, not covered by LoRA)
            torch.save(
                unwrapped_peft_unet.base_model.model.conv_in.state_dict(),
                os.path.join(save_path, "conv_in.pt"),
            )

            # MeasurementEncoder weights
            torch.save(
                accelerator.unwrap_model(measurement_encoder).state_dict(),
                os.path.join(save_path, "measurement_encoder.pt"),
            )


if __name__ == "__main__":
    main()
