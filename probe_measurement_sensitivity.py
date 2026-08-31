"""
Measurement sensitivity probe.

After a small training run, load a checkpoint and check whether the
MeasurementEncoder token actually moves UNet outputs.  Two tests:

  1. Sweep: vary one measurement dimension across [-3σ, +3σ] in 7 steps,
     record mean absolute UNet-output change relative to the baseline
     (mean measurements).  Prints a table and flags dead dimensions.

  2. Magnitude: compare the full output difference when all measurements
     are shifted to +2σ vs -2σ.  If this is near zero the token is ignored.

Usage:
    python probe_measurement_sensitivity.py \
        --checkpoint result/checkpoint-200 \
        --pretrained_model yisol/IDM-VTON \
        --data_dir gs://ma-idm-vton-data/datasets/fit-mini \
        --device cuda

The script uses a single batch from the test split, runs the UNet at a fixed
low-noise timestep (t=50), and does not generate full images.
"""

import argparse
import os
import torch
import torch.nn.functional as F
from peft import PeftModel

from diffusers import AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection

from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.measurement_encoder import MeasurementEncoder
from src.fit_dataset import FITDatasetWithMeasurements
from ip_adapter.ip_adapter import Resampler
from transformers import CLIPVisionModelWithProjection


MEASUREMENT_NAMES = [
    "body_bust", "body_height", "body_hips", "body_waist",
    "garment_bust", "garment_length", "garment_sleeve_length",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to checkpoint directory (contains lora/, conv_in.pt, measurement_encoder.pt)")
    p.add_argument("--pretrained_ip_adapter_path", type=str, default=None,
                            help="Path to IP-Adapter .bin. Defaults to ip_adapter/ip-adapter-plus_sdxl_vit-h.bin inside --pretrained_model_name_or_path.")
    p.add_argument("--pretrained_model", type=str, default="yisol/IDM-VTON")
    p.add_argument("--data_dir", type=str, default="gs://ma-idm-vton-data/datasets/fit-mini")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sigma_steps", type=int, default=7,
                   help="Number of σ steps in the sweep (odd number centred on 0)")
    p.add_argument("--sigma_range", type=float, default=3.0)
    p.add_argument("--fixed_timestep", type=int, default=50,
                   help="Diffusion timestep to use for probing (low = less noisy, more informative)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def unet_output(
    unet, unet_encoder, image_proj_model, measurement_encoder,
    batch, measurements, noise_scheduler, text_encoders, tokenizers,
    image_encoder, vae, device, dtype, fixed_t,
):
    """Single UNet forward at fixed_t with the given measurement vector."""
    tokenizer, tokenizer_2 = tokenizers
    text_encoder, text_encoder_2 = text_encoders

    # --- encode text ---
    ids1 = tokenizer(batch["text_prompts"], max_length=tokenizer.model_max_length,
                     padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
    ids2 = tokenizer_2(batch["text_prompts"], max_length=tokenizer_2.model_max_length,
                       padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
    enc1 = text_encoder(ids1, output_hidden_states=True)
    enc2 = text_encoder_2(ids2, output_hidden_states=True)
    text_embeds = enc1.hidden_states[-2]
    text_embeds_2 = enc2.hidden_states[-2]
    encoder_hidden_states = torch.cat([text_embeds, text_embeds_2], dim=-1)  # [B, 77, 2048]

    # --- measurement token ---
    m_token = measurement_encoder(measurements.to(device, dtype=torch.float32)).to(dtype)  # [B, 1, 2048]
    encoder_hidden_states = torch.cat([encoder_hidden_states, m_token], dim=1)  # [B, 78, 2048]

    # --- encode images to latents ---
    latent = vae.encode(batch["person_image"].to(device, dtype=vae.dtype)).latent_dist.sample() * vae.config.scaling_factor
    latent = latent.to(dtype)
    masked_latent = vae.encode(batch["masked_person"].to(device, dtype=vae.dtype)).latent_dist.sample() * vae.config.scaling_factor
    masked_latent = masked_latent.to(dtype)
    densepose_latent = vae.encode(batch["pose"].to(device, dtype=vae.dtype)).latent_dist.sample() * vae.config.scaling_factor
    densepose_latent = densepose_latent.to(dtype)
    cloth_latent = vae.encode(batch["garment_image"].to(device, dtype=vae.dtype)).latent_dist.sample() * vae.config.scaling_factor
    cloth_latent = cloth_latent.to(dtype)

    B = latent.shape[0]
    mask = F.interpolate(batch["mask"].to(device, dtype=dtype), size=(latent.shape[-2], latent.shape[-1]))

    torch.manual_seed(42)
    noise = torch.randn_like(latent)
    timesteps = torch.full((B,), fixed_t, device=device, dtype=torch.long)
    noisy = noise_scheduler.add_noise(latent, noise, timesteps)
    latent_model_input = torch.cat([noisy, mask, masked_latent, densepose_latent], dim=1)

    # --- IP-Adapter image embeds ---
    img_emb = batch["garment_image_clip"].to(device, dtype=dtype)
    if img_emb.dim() == 5:
        img_emb = img_emb.squeeze(1)
    clip_img_emb = image_encoder(img_emb, output_hidden_states=True).hidden_states[-2]
    ip_tokens = image_proj_model(clip_img_emb)

    # --- SDXL time/size ids ---
    orig_size = (batch["person_image"].shape[-2], batch["person_image"].shape[-1])
    add_time_ids = torch.tensor(
        [list(orig_size) + [0, 0] + list(orig_size)], device=device, dtype=dtype
    ).repeat(B, 1)
    pooled = enc2.text_embeds if hasattr(enc2, "text_embeds") else enc2.pooler_output
    unet_added = {"text_embeds": pooled, "time_ids": add_time_ids, "image_embeds": ip_tokens}

    # --- GarmentNet reference features ---
    ids1_c = tokenizer(batch["text_prompts_cloth"], max_length=tokenizer.model_max_length,
                       padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
    ids2_c = tokenizer_2(batch["text_prompts_cloth"], max_length=tokenizer_2.model_max_length,
                         padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
    enc1_c = text_encoder(ids1_c, output_hidden_states=True)
    enc2_c = text_encoder_2(ids2_c, output_hidden_states=True)
    text_cloth = torch.cat([enc1_c.hidden_states[-2], enc2_c.hidden_states[-2]], dim=-1)
    _, reference_features = unet_encoder(cloth_latent, timesteps, text_cloth, return_dict=False)
    reference_features = list(reference_features)

    out = unet(
        latent_model_input, timesteps, encoder_hidden_states,
        added_cond_kwargs=unet_added,
        garment_features=reference_features,
    ).sample
    return out


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.float16 if "cuda" in str(device) else torch.float32
    torch.manual_seed(args.seed)

    print(f"Loading models from {args.pretrained_model} …")

    tokenizer   = CLIPTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer_2")
    text_encoder   = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder").to(device, dtype=dtype)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model, subfolder="text_encoder_2").to(device, dtype=dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae").to(device, dtype=dtype)

    image_encoder_path = os.path.join(args.pretrained_model, "image_encoder") \
        if not os.path.isabs(args.pretrained_model) else os.path.join(args.pretrained_model, "image_encoder")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(device, dtype=dtype)

    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet",
                                                low_cpu_mem_usage=False, device_map=None)
    unet.config.encoder_hid_dim = image_encoder.config.hidden_size
    unet.config.encoder_hid_dim_type = "ip_image_proj"

    # Load IP-Adapter
    if args.pretrained_ip_adapter_path:
        ip_bin = args.pretrained_ip_adapter_path
    else:
        ip_bin = os.path.join(args.pretrained_model, "ip_adapter", "ip-adapter-plus_sdxl_vit-h.bin")
    ip_state = torch.load(ip_bin, map_location="cpu")
    import torch.nn as nn
    adapter_modules = nn.ModuleList(unet.attn_processors.values())
    adapter_modules.load_state_dict(ip_state["ip_adapter"], strict=True)

    image_proj_model = Resampler(
        dim=1280, depth=4, dim_head=64, heads=20,
        num_queries=16, embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim, ff_mult=4,
    ).to(device, dtype=dtype)
    image_proj_model.load_state_dict(ip_state["image_proj"])

    unet_encoder = UNet2DConditionModel_ref.from_pretrained(
        args.pretrained_model, subfolder="unet_encoder",
        low_cpu_mem_usage=False, device_map=None,
    )

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint} …")
    unet = PeftModel.from_pretrained(unet, os.path.join(args.checkpoint, "lora"))
    unet = unet.merge_and_unload()
    conv_in_state = torch.load(os.path.join(args.checkpoint, "conv_in.pt"), map_location="cpu")
    unet.conv_in.load_state_dict(conv_in_state)

    cross_attn_dim = unet.config.cross_attention_dim
    measurement_encoder = MeasurementEncoder(num_measurements=7, hidden_dim=256, output_dim=cross_attn_dim)
    menc_state = torch.load(os.path.join(args.checkpoint, "measurement_encoder.pt"), map_location="cpu")
    measurement_encoder.load_state_dict(menc_state)

    for m in [unet, unet_encoder, image_proj_model, text_encoder, text_encoder_2, image_encoder, vae]:
        m.to(device, dtype=dtype)
        m.eval()
    measurement_encoder.to(device, dtype=torch.float32)
    measurement_encoder.eval()

    # Load one test batch
    print("Loading test data …")
    from torch.utils.data import DataLoader
    dataset = FITDatasetWithMeasurements(data_root=args.data_dir, phase="test", size=(args.height, args.width))
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    # Baseline measurements (batch as-is)
    m_base = batch["measurements"].to(device, dtype=torch.float32)  # [B, 7], already normalised

    print("\nComputing baseline output …")
    out_base = unet_output(
        unet, unet_encoder, image_proj_model, measurement_encoder,
        batch, m_base, noise_scheduler,
        (text_encoder, text_encoder_2), (tokenizer, tokenizer_2),
        image_encoder, vae, device, dtype, args.fixed_timestep,
    )

    # ── Test 1: per-dimension sweep ──────────────────────────────────────────
    print("\n── Test 1: per-dimension sensitivity sweep ─────────────────────────────")
    print(f"{'Dim':<28}  {'max|Δ|':>9}  {'mean|Δ|':>9}  {'relative%':>10}")
    base_mag = out_base.abs().mean().item()
    sigmas = torch.linspace(-args.sigma_range, args.sigma_range, args.sigma_steps)
    for dim_idx, dim_name in enumerate(MEASUREMENT_NAMES):
        max_diff = 0.0
        mean_diff = 0.0
        for sigma in sigmas:
            if sigma.item() == 0.0:
                continue
            m_perturbed = m_base.clone()
            m_perturbed[:, dim_idx] += sigma.item()
            out_p = unet_output(
                unet, unet_encoder, image_proj_model, measurement_encoder,
                batch, m_perturbed, noise_scheduler,
                (text_encoder, text_encoder_2), (tokenizer, tokenizer_2),
                image_encoder, vae, device, dtype, args.fixed_timestep,
            )
            diff = (out_p - out_base).abs().mean().item()
            max_diff = max(max_diff, diff)
            mean_diff += diff
        mean_diff /= max(len(sigmas) - 1, 1)
        rel = 100.0 * mean_diff / (base_mag + 1e-8)
        flag = "  ← DEAD?" if mean_diff < 1e-5 else ""
        print(f"  {dim_name:<26}  {max_diff:9.6f}  {mean_diff:9.6f}  {rel:9.3f}%{flag}")

    # ── Test 2: global +2σ vs -2σ magnitude ──────────────────────────────────
    print("\n── Test 2: all dims +2σ vs -2σ ─────────────────────────────────────────")
    m_plus  = m_base + 2.0
    m_minus = m_base - 2.0
    out_plus  = unet_output(
        unet, unet_encoder, image_proj_model, measurement_encoder,
        batch, m_plus, noise_scheduler,
        (text_encoder, text_encoder_2), (tokenizer, tokenizer_2),
        image_encoder, vae, device, dtype, args.fixed_timestep,
    )
    out_minus = unet_output(
        unet, unet_encoder, image_proj_model, measurement_encoder,
        batch, m_minus, noise_scheduler,
        (text_encoder, text_encoder_2), (tokenizer, tokenizer_2),
        image_encoder, vae, device, dtype, args.fixed_timestep,
    )
    global_diff = (out_plus - out_minus).abs().mean().item()
    rel_global  = 100.0 * global_diff / (base_mag + 1e-8)
    print(f"  mean|out(+2σ) - out(-2σ)| = {global_diff:.6f}  ({rel_global:.3f}% of baseline magnitude)")
    if global_diff < 1e-5:
        print("  WARNING: measurement token appears to be completely ignored.")
    elif rel_global < 0.1:
        print("  WARNING: measurement effect is very small (<0.1% of baseline). Token may be near-ignored.")
    else:
        print("  OK: measurements have a detectable effect on UNet output.")


if __name__ == "__main__":
    main()
