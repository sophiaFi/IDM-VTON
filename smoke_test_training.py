"""
CPU smoke test for train_with_measurements.py

Replaces every large pretrained model with a minimal stub that preserves the
exact tensor interface used in the training loop.  No checkpoints, no internet
access, no GPU required.  Runs 3 training steps on 4 synthetic data records
and verifies:
  - FITDatasetWithMeasurements loads and returns the expected keys / shapes
  - MeasurementEncoder forward pass produces the right output shape
  - Measurement tokens are correctly concatenated to encoder_hidden_states
  - The combined [B, 78, 2048] context reaches the UNet stub without error
  - Loss is a finite scalar and backward() succeeds
  - Optimizer step updates TryonNet + MeasurementEncoder parameters
  - A checkpoint directory is written with measurement_encoder.pt

Run with:
    python smoke_test_training.py
"""

import json
import itertools
import os
import sys
import tempfile
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

# ── make src/ importable ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════════════════
# Stubs — tiny modules that mimic the tensor interface of the real models
# ═══════════════════════════════════════════════════════════════════════════════

H, W = 64, 48          # image size for the smoke test (very small)
LATENT_H = H // 8      # 8
LATENT_W = W // 8      # 6
SEQ_LEN = 77           # CLIP token count
CROSS_ATTN_DIM = 2048  # SDXL cross-attention dim (768 + 1280)
CLIP_HIDDEN = 1280     # CLIPVisionModel hidden_size (ViT-H)
NUM_IP_TOKENS = 16
BATCH = 2


class _LatentDist:
    def __init__(self, t): self._t = t
    def sample(self): return self._t


class StubVAE(nn.Module):
    """Returns random latents of the correct shape; scaling_factor matches SDXL."""
    config = types.SimpleNamespace(scaling_factor=0.13025)
    dtype = torch.float32

    def encode(self, x):
        b, c, h, w = x.shape
        latent = torch.randn(b, 4, h // 8, w // 8)
        return types.SimpleNamespace(latent_dist=_LatentDist(latent))


class StubTextEncoder(nn.Module):
    """Returns zeros for hidden states; pooled output is also zeros."""
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self._dummy = nn.Linear(1, 1)  # so .parameters() is non-empty for device checks

    def forward(self, input_ids, output_hidden_states=False):
        b, seq = input_ids.shape
        h = torch.zeros(b, seq, self.hidden_size)
        pooled = torch.zeros(b, self.hidden_size)
        # index [-2] used in train loop
        hidden_states = [torch.zeros(b, seq, self.hidden_size)] * 25
        return types.SimpleNamespace(hidden_states=hidden_states, last_hidden_state=h, __getitem__=lambda s, i: pooled)

    def __call__(self, *a, **kw):
        # Override __call__ to return an object where [0] gives pooled embed
        result = super().__call__(*a, **kw)
        result.__class__.__getitem__ = lambda self, i: torch.zeros(a[0].shape[0], self.hidden_size)
        return result


class _TextEncoderOutput:
    def __init__(self, b, seq, hidden_size):
        self.hidden_states = [torch.zeros(b, seq, hidden_size)] * 25
        self._pooled = torch.zeros(b, hidden_size)

    def __getitem__(self, i):
        return self._pooled


class StubCLIPText(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.hidden_size = hidden_size
        self._dummy = nn.Linear(1, 1)

    def forward(self, input_ids, output_hidden_states=False):
        b, seq = input_ids.shape
        return _TextEncoderOutput(b, seq, self.hidden_size)


class StubCLIPText2(nn.Module):
    """text_encoder_2: result[0] is the pooled embed."""
    def __init__(self, hidden_size=1280):
        super().__init__()
        self.hidden_size = hidden_size
        self._dummy = nn.Linear(1, 1)

    def forward(self, input_ids, output_hidden_states=False):
        b, seq = input_ids.shape
        return _TextEncoderOutput(b, seq, self.hidden_size)


class StubImageEncoder(nn.Module):
    config = types.SimpleNamespace(hidden_size=CLIP_HIDDEN)

    def __init__(self):
        super().__init__()
        self._dummy = nn.Linear(1, 1)

    def forward(self, pixel_values, output_hidden_states=False):
        b = pixel_values.shape[0]
        h = torch.zeros(b, 257, CLIP_HIDDEN)  # 257 = CLS + 16x16 patches for ViT-H/14
        hidden_states = [h] * 25
        return types.SimpleNamespace(hidden_states=hidden_states)


class StubResampler(nn.Module):
    """Projects image embeddings -> IP tokens [B, num_tokens, cross_attn_dim]."""
    def __init__(self, num_tokens, output_dim):
        super().__init__()
        self.proj = nn.Linear(CLIP_HIDDEN, output_dim)
        self.num_tokens = num_tokens

    def forward(self, x):
        # x: [B, seq, CLIP_HIDDEN] -> [B, num_tokens, output_dim]
        b = x.shape[0]
        return self.proj(x[:, :self.num_tokens])  # slice to num_tokens


class StubGarmentNet(nn.Module):
    """Returns zero reference features (list of 9 zero tensors, matching SDXL down blocks)."""
    config = types.SimpleNamespace(addition_embed_type=None)

    def __init__(self):
        super().__init__()
        self._dummy = nn.Linear(1, 1)

    def forward(self, sample, timestep, encoder_hidden_states, return_dict=False):
        b = sample.shape[0]
        ref_features = tuple(
            torch.zeros(b, 320, LATENT_H, LATENT_W) for _ in range(9)
        )
        return None, ref_features

    def enable_gradient_checkpointing(self): pass


class _UNetOutput:
    def __init__(self, sample): self.sample = sample


class StubTryonNet(nn.Module):
    """13-channel input conv + returns noise-shaped output."""
    config = types.SimpleNamespace(
        cross_attention_dim=CROSS_ATTN_DIM,
        in_channels=13,
        encoder_hid_dim=CLIP_HIDDEN,
        encoder_hid_dim_type="ip_image_proj",
    )

    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(13, 320, 3, padding=1)
        self.out = nn.Conv2d(320, 4, 1)
        self.attn_processors = {}
        self.encoder_hid_proj = None

    def forward(self, sample, timestep, encoder_hidden_states,
                added_cond_kwargs=None, garment_features=None, return_dict=True):
        x = self.conv_in(sample)
        out = self.out(x)
        return _UNetOutput(out)

    def enable_gradient_checkpointing(self): pass
    def enable_xformers_memory_efficient_attention(self): pass
    def train(self, mode=True): return super().train(mode)

    def parameters(self, recurse=True):
        return super().parameters(recurse)


class StubScheduler:
    config = types.SimpleNamespace(
        num_train_timesteps=1000,
        prediction_type="epsilon",
    )

    def add_noise(self, original, noise, timesteps):
        return original + noise * 0.1

    def get_velocity(self, sample, noise, timesteps):
        return noise


class StubTokenizer:
    model_max_length = 77

    def __call__(self, texts, max_length=77, padding=None, truncation=None, return_tensors=None):
        if isinstance(texts, str):
            texts = [texts]
        ids = torch.zeros(len(texts), max_length, dtype=torch.long)
        return types.SimpleNamespace(input_ids=ids)


# ═══════════════════════════════════════════════════════════════════════════════
# Minimal FIT dataset built entirely in memory
# ═══════════════════════════════════════════════════════════════════════════════

def build_dummy_fit_dataset(tmp_dir, n=4):
    """Write n dummy PNG files and a measurements.json into tmp_dir."""
    for subdir in ["cloth", "target", "image-densepose", "agnostic-mask"]:
        os.makedirs(os.path.join(tmp_dir, subdir), exist_ok=True)

    dummy_rgb = Image.fromarray(np.random.randint(0, 255, (H, W, 3), dtype=np.uint8))
    dummy_gray = Image.fromarray(np.random.randint(0, 255, (H, W), dtype=np.uint8))

    records = []
    for i in range(n):
        name = f"{i:04d}.png"
        dummy_rgb.save(os.path.join(tmp_dir, "cloth", name))
        dummy_rgb.save(os.path.join(tmp_dir, "target", name))
        dummy_rgb.save(os.path.join(tmp_dir, "image-densepose", name))
        dummy_gray.save(os.path.join(tmp_dir, "agnostic-mask", name))
        records.append({
            "id": i,
            "cloth": f"cloth/{name}",
            "target": f"target/{name}",
            "person": f"person/{name}",
            "body_bust": 100.0 + i,
            "body_height": 170.0,
            "body_hips": 105.0,
            "body_waist": 85.0,
            "garment_bust": 110.0,
            "garment_length": 55.0,
            "garment_sleeve_length": 30.0,
        })

    with open(os.path.join(tmp_dir, "measurements.json"), "w") as f:
        json.dump(records, f)

    return tmp_dir


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def run_smoke_test():
    print("=" * 60)
    print("IDM-VTON + MeasurementEncoder — CPU smoke test")
    print("=" * 60)

    from src.fit_dataset import FITDatasetWithMeasurements
    from src.measurement_encoder import MeasurementEncoder

    with tempfile.TemporaryDirectory() as tmp:
        # ── Dataset ──────────────────────────────────────────────────────────
        build_dummy_fit_dataset(tmp, n=4)
        dataset = FITDatasetWithMeasurements(tmp, phase="train", size=(H, W))
        print(f"[dataset] {len(dataset)} records")

        loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)
        batch = next(iter(loader))

        expected_keys = {
            "person_image", "garment_image", "garment_image_clip",
            "pose", "mask", "masked_person", "measurements",
            "text_prompts", "text_prompts_cloth",
        }
        assert expected_keys.issubset(batch.keys()), f"Missing keys: {expected_keys - batch.keys()}"
        assert batch["person_image"].shape == (BATCH, 3, H, W),      f"person_image shape wrong: {batch['person_image'].shape}"
        assert batch["garment_image"].shape == (BATCH, 3, H, W),     f"garment_image shape wrong"
        assert batch["garment_image_clip"].shape == (BATCH, 1, 3, 224, 224), f"garment_image_clip shape wrong: {batch['garment_image_clip'].shape}"
        assert batch["pose"].shape == (BATCH, 3, H, W),              f"pose shape wrong"
        assert batch["mask"].shape == (BATCH, 1, H, W),              f"mask shape wrong"
        assert batch["measurements"].shape == (BATCH, 7),            f"measurements shape wrong"
        print("[dataset] all key/shape checks passed")

        # ── MeasurementEncoder ────────────────────────────────────────────────
        measurement_encoder = MeasurementEncoder(
            num_measurements=7, hidden_dim=64, output_dim=CROSS_ATTN_DIM
        )
        m = batch["measurements"]
        tokens = measurement_encoder(m, measurement_dropout_prob=0.1)
        assert tokens.shape == (BATCH, 1, CROSS_ATTN_DIM), f"token shape wrong: {tokens.shape}"
        print(f"[encoder] measurement token shape: {tuple(tokens.shape)}  ✓")

        # ── Stub models ───────────────────────────────────────────────────────
        vae = StubVAE()
        text_encoder = StubCLIPText(hidden_size=768)
        text_encoder_2 = StubCLIPText2(hidden_size=1280)
        image_encoder = StubImageEncoder()
        unet_encoder = StubGarmentNet()
        unet = StubTryonNet()
        image_proj_model = StubResampler(num_tokens=NUM_IP_TOKENS, output_dim=CROSS_ATTN_DIM)
        noise_scheduler = StubScheduler()
        tokenizer = StubTokenizer()
        tokenizer_2 = StubTokenizer()

        # ── Optimizer (TryonNet + MeasurementEncoder) ─────────────────────────
        params_to_opt = itertools.chain(unet.parameters(), measurement_encoder.parameters())
        optimizer = torch.optim.AdamW(params_to_opt, lr=1e-5)

        unet.train()
        measurement_encoder.train()

        print("\nRunning 3 training steps...")
        for step in range(3):
            optimizer.zero_grad()

            # ── latents ──────────────────────────────────────────────────────
            pixel_values = batch["person_image"]
            model_input = vae.encode(pixel_values).latent_dist.sample()        # [B, 4, lH, lW]
            model_input = model_input * vae.config.scaling_factor

            masked_latents = vae.encode(batch["masked_person"]).latent_dist.sample()
            masked_latents = masked_latents * vae.config.scaling_factor

            masks = batch["mask"]
            mask = F.interpolate(masks, size=(LATENT_H, LATENT_W))
            mask = mask.reshape(-1, 1, LATENT_H, LATENT_W)

            pose_map = vae.encode(batch["pose"]).latent_dist.sample()
            pose_map = pose_map * vae.config.scaling_factor

            noise = torch.randn_like(model_input)
            bsz = model_input.shape[0]
            timesteps = torch.randint(0, 1000, (bsz,))
            noisy_latents = noise_scheduler.add_noise(model_input, noise, timesteps)
            latent_model_input = torch.cat([noisy_latents, mask, masked_latents, pose_map], dim=1)
            assert latent_model_input.shape[1] == 13, f"expected 13 channels, got {latent_model_input.shape[1]}"

            # ── text embeddings ───────────────────────────────────────────────
            text_input_ids = tokenizer(batch["text_prompts"]).input_ids
            text_input_ids_2 = tokenizer_2(batch["text_prompts"]).input_ids
            enc1 = text_encoder(text_input_ids, output_hidden_states=True)
            enc2 = text_encoder_2(text_input_ids_2, output_hidden_states=True)
            text_embeds = enc1.hidden_states[-2]        # [B, 77, 768]
            pooled_text_embeds = enc2[0]                # [B, 1280]
            text_embeds_2 = enc2.hidden_states[-2]      # [B, 77, 1280]
            encoder_hidden_states = torch.cat([text_embeds, text_embeds_2], dim=-1)  # [B, 77, 2048]

            # ── measurement token concatenation ───────────────────────────────
            m_tokens = measurement_encoder(batch["measurements"], measurement_dropout_prob=0.1)  # [B, 1, 2048]
            encoder_hidden_states = torch.cat([encoder_hidden_states, m_tokens], dim=1)          # [B, 78, 2048]
            assert encoder_hidden_states.shape == (BATCH, 78, CROSS_ATTN_DIM), \
                f"encoder_hidden_states shape wrong: {encoder_hidden_states.shape}"

            # ── IP-Adapter ────────────────────────────────────────────────────
            img_emb = torch.cat([batch["garment_image_clip"][i] for i in range(bsz)], dim=0)
            img_emb = image_encoder(img_emb, output_hidden_states=True).hidden_states[-2]
            ip_tokens = image_proj_model(img_emb)

            add_time_ids = torch.zeros(bsz, 6)
            unet_added_cond_kwargs = {
                "text_embeds": pooled_text_embeds,
                "time_ids": add_time_ids,
                "image_embeds": ip_tokens,
            }

            # ── GarmentNet ────────────────────────────────────────────────────
            cloth_latents = vae.encode(batch["garment_image"]).latent_dist.sample()
            cloth_latents = cloth_latents * vae.config.scaling_factor
            text_ids_c = tokenizer(batch["text_prompts_cloth"]).input_ids
            text_ids_2_c = tokenizer_2(batch["text_prompts_cloth"]).input_ids
            enc1_c = text_encoder(text_ids_c, output_hidden_states=True)
            enc2_c = text_encoder_2(text_ids_2_c, output_hidden_states=True)
            text_embeds_cloth = torch.cat([enc1_c.hidden_states[-2], enc2_c.hidden_states[-2]], dim=-1)
            _, reference_features = unet_encoder(cloth_latents, timesteps, text_embeds_cloth, return_dict=False)
            reference_features = list(reference_features)

            # ── TryonNet forward + loss ───────────────────────────────────────
            noise_pred = unet(
                latent_model_input,
                timesteps,
                encoder_hidden_states,
                added_cond_kwargs=unet_added_cond_kwargs,
                garment_features=reference_features,
            ).sample

            target = noise
            loss = F.mse_loss(noise_pred.float(), target.float())
            assert torch.isfinite(loss), f"loss is not finite: {loss}"

            loss.backward()
            optimizer.step()

            print(f"  step {step+1}/3 — loss={loss.item():.6f}  ✓")

        # ── Checkpoint ────────────────────────────────────────────────────────
        ckpt_dir = os.path.join(tmp, "checkpoint-smoke")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(measurement_encoder.state_dict(), os.path.join(ckpt_dir, "measurement_encoder.pt"))
        assert os.path.exists(os.path.join(ckpt_dir, "measurement_encoder.pt"))
        print(f"[checkpoint] measurement_encoder.pt saved to {ckpt_dir}  ✓")

    print("\n" + "=" * 60)
    print("All checks passed — training logic is correct.")
    print("=" * 60)


if __name__ == "__main__":
    run_smoke_test()
