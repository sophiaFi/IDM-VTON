"""
Hyperparameter optimisation for IDM-VTON + MeasurementEncoder + FiLM extension.

Differences from train_hpo.py (base extension):
- Custom LoRAConv2d on ResNet convs instead of PEFT LoRA on attention layers.
- Measurements injected via FiLM modulation (γ/β on ResNet norm2 output),
  not by appending a token to the cross-attention sequence.
- FiLMGeneratorSet is a trainable component alongside MeasurementEncoder.
- lora_alpha_ratio is dropped (PEFT-specific); lora_rank now controls ResNet convs.

Usage (called from vertex_hpo_job_film.yaml):
  python train_hpo_film.py \
      --data_dir /tmp/fit-mini \
      --pretrained_model_name_or_path /tmp/yisol-IDM-VTON \
      --pretrained_ip_adapter_path /tmp/ckpt/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin \
      --output_dir /tmp/hpo_output \
      --n_trials 20 \
      --max_proxy_steps 1000 \
      --mixed_precision bf16 \
      --gradient_checkpointing \
      --gcs_study_bucket idm-vton-bucket \
      --gcs_study_prefix film_hpo/study.db
"""

import argparse
import gc
import itertools
import json
import math
import os
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    SegformerForSemanticSegmentation,
)

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

from ip_adapter.ip_adapter import Resampler
from src.fit_dataset import FITDatasetWithMeasurements
from src.film_generator import FiLMGeneratorSet
from src.measurement_encoder import MeasurementEncoder
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel


# ---------------------------------------------------------------------------
# FiLM infrastructure (mirrors train_with_film.py)
# ---------------------------------------------------------------------------

class _FiLMContext:
    embedding: Optional[torch.Tensor] = None


class LoRAConv2d(torch.nn.Module):
    def __init__(self, conv: torch.nn.Conv2d, rank: int = 4) -> None:
        super().__init__()
        self.conv = conv
        self.conv.requires_grad_(False)
        C_in, C_out = conv.in_channels, conv.out_channels
        k, pad = conv.kernel_size[0], conv.padding[0]
        self.lora_A = torch.nn.Conv2d(C_in, rank, 1, bias=False)
        self.lora_B = torch.nn.Conv2d(rank, C_out, k, padding=pad, bias=False)
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.lora_B(self.lora_A(x))


def _apply_lora_to_resnets(resnets: list, rank: int) -> None:
    for resnet in resnets:
        resnet.conv1 = LoRAConv2d(resnet.conv1, rank=rank)
        resnet.conv2 = LoRAConv2d(resnet.conv2, rank=rank)
        if getattr(resnet, "conv_shortcut", None) is not None:
            resnet.conv_shortcut = LoRAConv2d(resnet.conv_shortcut, rank=rank)


def _get_target_resnets(unet) -> list:
    resnets = []
    resnets.extend(list(unet.down_blocks[-1].resnets))
    resnets.extend(list(unet.mid_block.resnets))
    resnets.extend(list(unet.up_blocks[0].resnets))
    return resnets


def _register_film_hooks(unet, film_generators: FiLMGeneratorSet, ctx: _FiLMContext) -> list:
    target_resnets = _get_target_resnets(unet)
    hooks = []
    for resnet, head in zip(target_resnets, film_generators.heads):
        def _make_hook(h):
            def _hook(module, inp, out):
                if ctx.embedding is None:
                    return out
                emb = ctx.embedding
                B_out = out.shape[0]
                B_emb = emb.shape[0]
                if B_out != B_emb:
                    emb = emb.repeat(B_out // B_emb, 1)
                gamma, beta = h(emb.to(device=out.device, dtype=torch.float32))
                gamma = gamma.to(dtype=out.dtype)
                beta = beta.to(dtype=out.dtype)
                return gamma[:, :, None, None] * out + beta[:, :, None, None]
            return _hook
        hooks.append(resnet.norm2.register_forward_hook(_make_hook(head)))
    return hooks


# ---------------------------------------------------------------------------
# Losses (identical to train_hpo.py)
# ---------------------------------------------------------------------------

_SEGFORMER_GARMENT_IDS = (4, 7)
_SEG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_SEG_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_SEG_SIZE = 512


def _boundary_loss(pred, gt):
    pool = torch.nn.functional.max_pool2d
    erode = lambda x: -torch.nn.functional.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
    boundary_pred = pool(pred, kernel_size=3, stride=1, padding=1) - erode(pred)
    boundary_gt   = pool(gt,   kernel_size=3, stride=1, padding=1) - erode(gt)
    n_boundary = (boundary_gt > 0.1).float().sum().clamp(min=1.0)
    return (boundary_pred - boundary_gt).abs().sum() / n_boundary


def _compute_fit_loss(noise_pred, noisy_latents, timesteps, gt_garment_mask,
                      noise_scheduler, prediction_type, vae, segmenter,
                      alpha_threshold=0.3, boundary_weight=0.5):
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(noisy_latents.device)
    alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(noisy_latents.dtype)
    keep = alpha_t.view(-1) > alpha_threshold
    if not keep.any():
        return torch.tensor(0.0, device=noisy_latents.device, requires_grad=False)

    noise_pred_k    = noise_pred[keep]
    noisy_latents_k = noisy_latents[keep]
    alpha_t_k       = alpha_t[keep]
    gt_mask         = gt_garment_mask[keep].clamp(0, 1).to(noisy_latents.dtype)

    sqrt_alpha = alpha_t_k.sqrt()
    sqrt_one_minus = (1.0 - alpha_t_k).sqrt()
    if prediction_type == "v_prediction":
        x0_pred = sqrt_alpha * noisy_latents_k - sqrt_one_minus * noise_pred_k
    elif prediction_type == "sample":
        x0_pred = noise_pred_k
    else:
        x0_pred = (noisy_latents_k - sqrt_one_minus * noise_pred_k) / sqrt_alpha.clamp(min=1e-8)

    decoded = vae.decode(
        (x0_pred / vae.config.scaling_factor).to(dtype=vae.dtype)
    ).sample.float()

    decoded_01 = (decoded.clamp(-1, 1) + 1.0) / 2.0
    seg_mean = _SEG_MEAN.to(decoded.device, dtype=decoded.dtype)
    seg_std  = _SEG_STD.to(decoded.device, dtype=decoded.dtype)
    seg_input_rs = F.interpolate(
        (decoded_01 - seg_mean) / seg_std,
        size=(_SEG_SIZE, _SEG_SIZE), mode="bilinear", align_corners=False,
    )
    logits = segmenter(
        pixel_values=seg_input_rs.to(dtype=next(segmenter.parameters()).dtype)
    ).logits.float()

    H, W = gt_mask.shape[-2], gt_mask.shape[-1]
    garment_logits = sum(logits[:, cls_id:cls_id+1] for cls_id in _SEGFORMER_GARMENT_IDS)
    soft_pred = F.interpolate(
        torch.sigmoid(garment_logits), size=(H, W), mode="bilinear", align_corners=False,
    )

    intersection = (soft_pred * gt_mask).sum(dim=(1, 2, 3))
    union = (soft_pred + gt_mask - soft_pred * gt_mask).sum(dim=(1, 2, 3)).clamp(min=1e-8)
    iou_loss = (1.0 - intersection / union).mean()
    return iou_loss + boundary_weight * _boundary_loss(soft_pred, gt_mask)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path", type=str, default="/tmp/yisol-IDM-VTON")
    p.add_argument("--pretrained_ip_adapter_path",    type=str, default=None)
    p.add_argument("--data_dir",    type=str, default="/tmp/fit-mini")
    p.add_argument("--output_dir",  type=str, default="/tmp/hpo_output")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width",  type=int, default=384)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--num_tokens", type=int, default=16)
    p.add_argument("--segmenter_model", type=str, default="mattmdjaga/segformer_b2_clothes")
    # HPO settings
    p.add_argument("--n_trials",          type=int,   default=20)
    p.add_argument("--max_proxy_steps",   type=int,   default=1000,
                   help="Training steps per trial (proxy budget).")
    p.add_argument("--val_batches",       type=int,   default=20,
                   help="Validation batches used to compute the trial metric.")
    p.add_argument("--pruner_min_resource", type=int, default=200)
    p.add_argument("--seed",              type=int,   default=0)
    p.add_argument("--val_split",         type=float, default=0.1)
    # GCS study persistence
    p.add_argument("--gcs_study_bucket", type=str, default=None)
    p.add_argument("--gcs_study_prefix", type=str, default="film_hpo/study.db")
    return p.parse_args()


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------

def _gcs_download(bucket_name, blob_name, local_path):
    try:
        from google.cloud import storage as gcs
    except ImportError:
        return False
    client = gcs.Client()
    blob = client.bucket(bucket_name).blob(blob_name)
    if not blob.exists():
        return False
    blob.download_to_filename(local_path)
    return True


def _gcs_upload(local_path, bucket_name, blob_name):
    try:
        from google.cloud import storage as gcs
    except ImportError:
        return
    gcs.Client().bucket(bucket_name).blob(blob_name).upload_from_filename(local_path)


# ---------------------------------------------------------------------------
# Frozen components (loaded once for all trials)
# ---------------------------------------------------------------------------

class FrozenComponents:
    def __init__(self, args, device, weight_dtype):
        base = args.pretrained_model_name_or_path

        self.noise_scheduler = DDPMScheduler.from_pretrained(
            base, subfolder="scheduler", rescale_betas_zero_snr=True
        )
        self.tokenizer   = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer_2")

        self.text_encoder = CLIPTextModel.from_pretrained(
            base, subfolder="text_encoder"
        ).to(device, dtype=weight_dtype).eval()
        self.text_encoder.requires_grad_(False)

        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            base, subfolder="text_encoder_2"
        ).to(device, dtype=weight_dtype).eval()
        self.text_encoder_2.requires_grad_(False)

        self.vae = AutoencoderKL.from_pretrained(
            base, subfolder="vae", torch_dtype=torch.float16
        ).to(device).eval()
        self.vae.requires_grad_(False)

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base, subfolder="image_encoder"
        ).to(device, dtype=weight_dtype).eval()
        self.image_encoder.requires_grad_(False)

        self.unet_encoder = UNet2DConditionModel_ref.from_pretrained(
            base, subfolder="unet_encoder"
        ).to(device, dtype=weight_dtype).eval()
        self.unet_encoder.config.addition_embed_type = None
        self.unet_encoder.config["addition_embed_type"] = None
        self.unet_encoder.requires_grad_(False)

        # Base TryonNet: load + prepare, then keep as CPU state dict for cheap per-trial reset.
        unet_base = UNet2DConditionModel.from_pretrained(
            base, subfolder="unet", low_cpu_mem_usage=False, device_map=None
        )
        unet_base.config.encoder_hid_dim = self.image_encoder.config.hidden_size
        unet_base.config.encoder_hid_dim_type = "ip_image_proj"
        unet_base.config["encoder_hid_dim"] = self.image_encoder.config.hidden_size
        unet_base.config["encoder_hid_dim_type"] = "ip_image_proj"

        ip_bin = (
            args.pretrained_ip_adapter_path
            or os.path.join(base, "ip_adapter", "ip-adapter-plus_sdxl_vit-h.bin")
        )
        state_dict = torch.load(ip_bin, map_location="cpu")
        adapter_modules = torch.nn.ModuleList(unet_base.attn_processors.values())
        adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        image_proj_model = Resampler(
            dim=self.image_encoder.config.hidden_size,
            depth=4, dim_head=64, heads=20,
            num_queries=args.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=unet_base.config.cross_attention_dim,
            ff_mult=4,
        )
        image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        unet_base.encoder_hid_proj = image_proj_model

        if unet_base.conv_in.in_channels == 9:
            conv_new = torch.nn.Conv2d(13, unet_base.conv_in.out_channels, 3, padding=1)
            torch.nn.init.zeros_(conv_new.weight)
            conv_new.weight.data[:, :9] = unet_base.conv_in.weight.data
            conv_new.bias.data = unet_base.conv_in.bias.data
            unet_base.conv_in = conv_new
            unet_base.config["in_channels"] = 13
            unet_base.config.in_channels = 13

        # Derive FiLM channel dims from the base UNet (before any LoRA wrapping)
        self.film_channel_dims = [r.norm2.num_channels for r in _get_target_resnets(unet_base)]

        self.unet_base_state_dict = {k: v.cpu() for k, v in unet_base.state_dict().items()}
        self.unet_base_config = unet_base.config
        self.cross_attn_dim = unet_base.config.cross_attention_dim
        del unet_base

        self.segmenter = SegformerForSemanticSegmentation.from_pretrained(
            args.segmenter_model
        ).to(device, dtype=torch.float32).eval()
        self.segmenter.requires_grad_(False)

        self.ip_state_dict = state_dict
        self.device = device
        self.weight_dtype = weight_dtype


# ---------------------------------------------------------------------------
# Per-trial UNet construction
# ---------------------------------------------------------------------------

def _build_trial_unet(frozen: FrozenComponents, lora_rank: int):
    """Reconstruct a LoRA-wrapped TryonNet from the frozen base state dict.

    Returns (unet, film_channel_dims). The Resampler is rebuilt explicitly because
    UNet2DConditionModel.from_config does not include encoder_hid_proj in its
    architecture, so load_state_dict(strict=False) would skip those weights.
    """
    unet = UNet2DConditionModel.from_config(frozen.unet_base_config)
    unet.load_state_dict(frozen.unet_base_state_dict, strict=False)

    image_proj_model = Resampler(
        dim=frozen.image_encoder.config.hidden_size,
        depth=4, dim_head=64, heads=20,
        num_queries=16,
        embedding_dim=frozen.image_encoder.config.hidden_size,
        output_dim=frozen.cross_attn_dim,
        ff_mult=4,
    )
    image_proj_model.load_state_dict(frozen.ip_state_dict["image_proj"], strict=True)
    unet.encoder_hid_proj = image_proj_model

    # Gradient checkpointing is intentionally disabled for HPO: FiLM hooks inject a
    # grad-tracked tensor (meas_emb from trainable measurement_encoder) into multiple
    # checkpointed segments, which triggers "tensor saved by autograd multiple times"
    # errors during recomputation. HPO trials are short (<=1000 steps, batch size 1-2)
    # so the memory saving is not needed.

    unet.requires_grad_(False)
    unet.conv_in.requires_grad_(True)

    if lora_rank > 0:
        _apply_lora_to_resnets(_get_target_resnets(unet), rank=lora_rank)
        for resnet in _get_target_resnets(unet):
            resnet.conv1.lora_A.requires_grad_(True)
            resnet.conv1.lora_B.requires_grad_(True)
            resnet.conv2.lora_A.requires_grad_(True)
            resnet.conv2.lora_B.requires_grad_(True)
            if getattr(resnet, "conv_shortcut", None) is not None:
                resnet.conv_shortcut.lora_A.requires_grad_(True)
                resnet.conv_shortcut.lora_B.requires_grad_(True)
            resnet.time_emb_proj.requires_grad_(True)
            resnet.norm1.requires_grad_(True)
            resnet.norm2.requires_grad_(True)

    return unet.to(frozen.device, dtype=frozen.weight_dtype)


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def make_objective(frozen: FrozenComponents, train_ds, val_ds, args):

    def objective(trial: optuna.Trial) -> float:
        lr                  = trial.suggest_float("lr", 1e-5, 2e-4, log=True)
        lora_rank           = trial.suggest_categorical("lora_rank", [4, 8, 16])
        fit_loss_w          = 0.3
        fit_alpha_threshold = trial.suggest_categorical("fit_loss_alpha_threshold", [0.2, 0.3, 0.5])
        menc_dropout        = trial.suggest_float("measurement_dropout", 0.0, 0.2)
        lr_scheduler_type   = trial.suggest_categorical("lr_scheduler", ["constant", "cosine"])
        batch_size          = trial.suggest_categorical("batch_size", [1, 2])

        torch.manual_seed(args.seed + trial.number)
        device = frozen.device

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=2
        )

        unet = _build_trial_unet(frozen, lora_rank)
        film_generators = FiLMGeneratorSet(
            input_dim=frozen.cross_attn_dim, channel_dims=frozen.film_channel_dims
        ).to(device)
        measurement_encoder = MeasurementEncoder(
            num_measurements=9, hidden_dim=256,
            output_dim=frozen.cross_attn_dim, dropout=menc_dropout, use_fourier=False,
        ).to(device)

        film_ctx = _FiLMContext()
        hooks = _register_film_hooks(unet, film_generators, film_ctx)

        trainable_params = list(itertools.chain(
            (p for p in unet.parameters() if p.requires_grad),
            film_generators.parameters(),
            measurement_encoder.parameters(),
        ))
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)

        if lr_scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.max_proxy_steps
            )
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

        unet.train()
        film_generators.train()
        measurement_encoder.train()

        global_step = 0
        train_iter = iter(train_loader)
        report_interval = max(1, args.pruner_min_resource // 2)

        while global_step < args.max_proxy_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            with torch.no_grad():
                pixel_values = batch["person_image"].to(dtype=frozen.vae.dtype, device=device)
                model_input = (frozen.vae.encode(pixel_values).latent_dist.sample()
                               * frozen.vae.config.scaling_factor).to(dtype=frozen.weight_dtype)
                masked_latents = (frozen.vae.encode(
                    batch["masked_person"].reshape(batch["person_image"].shape).to(dtype=frozen.vae.dtype, device=device)
                ).latent_dist.sample() * frozen.vae.config.scaling_factor).to(dtype=frozen.weight_dtype)
                pose_map = (frozen.vae.encode(
                    batch["pose"].to(dtype=frozen.vae.dtype, device=device)
                ).latent_dist.sample() * frozen.vae.config.scaling_factor).to(dtype=frozen.weight_dtype)

                masks = batch["mask"].to(device)
                mask = F.interpolate(masks, size=(args.height // 8, args.width // 8)).reshape(
                    -1, 1, args.height // 8, args.width // 8
                ).to(dtype=frozen.weight_dtype)

                bsz = model_input.shape[0]
                noise = torch.randn_like(model_input)
                timesteps = torch.randint(
                    0, frozen.noise_scheduler.config.num_train_timesteps, (bsz,), device=device
                )
                noisy_latents = frozen.noise_scheduler.add_noise(model_input, noise, timesteps)
                latent_model_input = torch.cat([noisy_latents, mask, masked_latents, pose_map], dim=1)

                def _encode_text(prompts):
                    ids = frozen.tokenizer(
                        prompts, max_length=frozen.tokenizer.model_max_length,
                        padding="max_length", truncation=True, return_tensors="pt",
                    ).input_ids.to(device)
                    ids2 = frozen.tokenizer_2(
                        prompts, max_length=frozen.tokenizer_2.model_max_length,
                        padding="max_length", truncation=True, return_tensors="pt",
                    ).input_ids.to(device)
                    enc1 = frozen.text_encoder(ids, output_hidden_states=True)
                    enc2 = frozen.text_encoder_2(ids2, output_hidden_states=True)
                    return torch.cat([enc1.hidden_states[-2], enc2.hidden_states[-2]], dim=-1), enc2[0]

                encoder_hidden_states, pooled_text_embeds = _encode_text(batch["text_prompts"])

                add_time_ids = torch.cat([
                    torch.tensor([[args.height, args.width, 0, 0, args.height, args.width]],
                                 device=device, dtype=frozen.weight_dtype)
                    for _ in range(bsz)
                ])

                image_embeds = torch.cat(
                    [batch["garment_image_clip"][i] for i in range(bsz)], dim=0
                ).to(device, dtype=frozen.weight_dtype)
                image_embeds = frozen.image_encoder(image_embeds, output_hidden_states=True).hidden_states[-2]
                ip_tokens = unet.encoder_hid_proj(image_embeds)

                unet_added = {
                    "text_embeds": pooled_text_embeds,
                    "time_ids": add_time_ids,
                    "image_embeds": ip_tokens,
                }

                cloth_values = (frozen.vae.encode(
                    batch["garment_image"].to(dtype=frozen.vae.dtype, device=device)
                ).latent_dist.sample() * frozen.vae.config.scaling_factor).to(dtype=frozen.weight_dtype)
                text_embeds_cloth, _ = _encode_text(batch["text_prompts_cloth"])
                _, reference_features = frozen.unet_encoder(
                    cloth_values, timesteps, text_embeds_cloth, return_dict=False
                )
                reference_features = list(reference_features)

            # Measurement embedding → FiLM context (outside no_grad so gradients flow)
            measurements = batch["measurements"].to(device, dtype=torch.float32)
            meas_emb = measurement_encoder(
                measurements, measurement_dropout_prob=menc_dropout
            ).squeeze(1)  # [B, cross_attn_dim]
            film_ctx.embedding = meas_emb.to(dtype=torch.float32)

            noise_pred = unet(
                latent_model_input, timesteps, encoder_hidden_states,
                added_cond_kwargs=unet_added,
                garment_features=reference_features,
            ).sample

            film_ctx.embedding = None

            if frozen.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif frozen.noise_scheduler.config.prediction_type == "v_prediction":
                target = frozen.noise_scheduler.get_velocity(model_input, noise, timesteps)
            else:
                target = model_input

            loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
            if fit_loss_w > 0:
                fl = _compute_fit_loss(
                    noise_pred.float(), noisy_latents.float(), timesteps,
                    batch["garment_mask"].to(device).float(),
                    frozen.noise_scheduler, frozen.noise_scheduler.config.prediction_type,
                    frozen.vae, frozen.segmenter,
                    alpha_threshold=fit_alpha_threshold, boundary_weight=0.5,
                )
                loss = loss + fit_loss_w * fl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % report_interval == 0:
                val_loss = _evaluate(
                    unet, measurement_encoder, film_generators, film_ctx,
                    frozen, val_loader, args, fit_loss_w, n_batches=5,
                    fit_alpha_threshold=fit_alpha_threshold,
                )
                trial.report(val_loss, step=global_step)
                if trial.should_prune():
                    for h in hooks:
                        h.remove()
                    del unet, measurement_encoder, film_generators, optimizer, scheduler, hooks
                    gc.collect()
                    torch.cuda.empty_cache()
                    raise optuna.TrialPruned()

        val_loss = _evaluate(
            unet, measurement_encoder, film_generators, film_ctx,
            frozen, val_loader, args, fit_loss_w, n_batches=args.val_batches,
            fit_alpha_threshold=fit_alpha_threshold,
        )
        trial.set_user_attr("lora_rank", lora_rank)

        for h in hooks:
            h.remove()
        del unet, measurement_encoder, film_generators, optimizer, scheduler, hooks
        gc.collect()
        torch.cuda.empty_cache()
        return val_loss

    return objective


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate(unet, measurement_encoder, film_generators, film_ctx: _FiLMContext,
              frozen: FrozenComponents, val_loader, args, fit_loss_w: float,
              n_batches: int, fit_alpha_threshold: float = 0.3) -> float:
    unet.eval()
    film_generators.eval()
    measurement_encoder.eval()
    device = frozen.device
    total_loss = 0.0
    n_seen = 0

    val_iter = iter(val_loader)
    for _ in range(n_batches):
        try:
            batch = next(val_iter)
        except StopIteration:
            break

        pixel_values = batch["person_image"].to(dtype=frozen.vae.dtype, device=device)
        model_input = (frozen.vae.encode(pixel_values).latent_dist.sample()
                       * frozen.vae.config.scaling_factor).to(dtype=frozen.weight_dtype)
        masked_latents = (frozen.vae.encode(
            batch["masked_person"].reshape(batch["person_image"].shape).to(dtype=frozen.vae.dtype, device=device)
        ).latent_dist.sample() * frozen.vae.config.scaling_factor).to(dtype=frozen.weight_dtype)
        pose_map = (frozen.vae.encode(
            batch["pose"].to(dtype=frozen.vae.dtype, device=device)
        ).latent_dist.sample() * frozen.vae.config.scaling_factor).to(dtype=frozen.weight_dtype)
        masks = batch["mask"].to(device)
        mask = F.interpolate(masks, size=(args.height // 8, args.width // 8)).reshape(
            -1, 1, args.height // 8, args.width // 8
        ).to(dtype=frozen.weight_dtype)

        bsz = model_input.shape[0]
        noise = torch.randn_like(model_input)
        timesteps = torch.randint(
            0, frozen.noise_scheduler.config.num_train_timesteps, (bsz,), device=device
        )
        noisy_latents = frozen.noise_scheduler.add_noise(model_input, noise, timesteps)
        latent_model_input = torch.cat([noisy_latents, mask, masked_latents, pose_map], dim=1)

        def _encode_text(prompts):
            ids = frozen.tokenizer(
                prompts, max_length=frozen.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt",
            ).input_ids.to(device)
            ids2 = frozen.tokenizer_2(
                prompts, max_length=frozen.tokenizer_2.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt",
            ).input_ids.to(device)
            enc1 = frozen.text_encoder(ids, output_hidden_states=True)
            enc2 = frozen.text_encoder_2(ids2, output_hidden_states=True)
            return torch.cat([enc1.hidden_states[-2], enc2.hidden_states[-2]], dim=-1), enc2[0]

        encoder_hidden_states, pooled_text_embeds = _encode_text(batch["text_prompts"])

        add_time_ids = torch.cat([
            torch.tensor([[args.height, args.width, 0, 0, args.height, args.width]],
                         device=device, dtype=frozen.weight_dtype)
            for _ in range(bsz)
        ])
        image_embeds = torch.cat(
            [batch["garment_image_clip"][i] for i in range(bsz)], dim=0
        ).to(device, dtype=frozen.weight_dtype)
        image_embeds = frozen.image_encoder(image_embeds, output_hidden_states=True).hidden_states[-2]
        ip_tokens = unet.encoder_hid_proj(image_embeds)

        unet_added = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids, "image_embeds": ip_tokens}

        cloth_values = (frozen.vae.encode(
            batch["garment_image"].to(dtype=frozen.vae.dtype, device=device)
        ).latent_dist.sample() * frozen.vae.config.scaling_factor).to(dtype=frozen.weight_dtype)
        text_embeds_cloth, _ = _encode_text(batch["text_prompts_cloth"])
        _, reference_features = frozen.unet_encoder(
            cloth_values, timesteps, text_embeds_cloth, return_dict=False
        )
        reference_features = list(reference_features)

        measurements = batch["measurements"].to(device, dtype=torch.float32)
        meas_emb = measurement_encoder(measurements, measurement_dropout_prob=0.0).squeeze(1)
        film_ctx.embedding = meas_emb.to(dtype=torch.float32)

        noise_pred = unet(
            latent_model_input, timesteps, encoder_hidden_states,
            added_cond_kwargs=unet_added,
            garment_features=reference_features,
        ).sample

        film_ctx.embedding = None

        if frozen.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif frozen.noise_scheduler.config.prediction_type == "v_prediction":
            target = frozen.noise_scheduler.get_velocity(model_input, noise, timesteps)
        else:
            target = model_input

        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
        if fit_loss_w > 0:
            fl = _compute_fit_loss(
                noise_pred.float(), noisy_latents.float(), timesteps,
                batch["garment_mask"].to(device).float(),
                frozen.noise_scheduler, frozen.noise_scheduler.config.prediction_type,
                frozen.vae, frozen.segmenter,
                alpha_threshold=fit_alpha_threshold,
            )
            loss = loss + fit_loss_w * fl

        total_loss += loss.item()
        n_seen += 1

    unet.train()
    film_generators.train()
    measurement_encoder.train()
    return total_loss / max(n_seen, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.mixed_precision, torch.float32)

    print("Loading frozen components (once for all trials)...")
    frozen = FrozenComponents(args, device, weight_dtype)

    full_train_ds = FITDatasetWithMeasurements(
        data_root=args.data_dir, phase="train", size=(args.height, args.width),
    )
    n_val   = max(1, int(len(full_train_ds) * args.val_split))
    n_train = len(full_train_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_train_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"Dataset: {len(train_ds)} train / {len(val_ds)} val")

    study_db = os.path.join(args.output_dir, "study.db")
    if args.gcs_study_bucket:
        print(f"Checking GCS for existing study at gs://{args.gcs_study_bucket}/{args.gcs_study_prefix} ...")
        if _gcs_download(args.gcs_study_bucket, args.gcs_study_prefix, study_db):
            print("Resumed existing study from GCS.")
        else:
            print("No existing study found — starting fresh.")

    storage = f"sqlite:///{study_db}"
    study = optuna.create_study(
        study_name="idm_vton_film_hpo",
        direction="minimize",
        sampler=TPESampler(seed=args.seed),
        pruner=HyperbandPruner(
            min_resource=args.pruner_min_resource,
            max_resource=args.max_proxy_steps,
            reduction_factor=3,
        ),
        storage=storage,
        load_if_exists=True,
    )

    def _after_trial(study, trial):
        if args.gcs_study_bucket and os.path.exists(study_db):
            _gcs_upload(study_db, args.gcs_study_bucket, args.gcs_study_prefix)
            print(f"Study checkpoint uploaded to gs://{args.gcs_study_bucket}/{args.gcs_study_prefix}")

    objective = make_objective(frozen, train_ds, val_ds, args)
    study.optimize(objective, n_trials=args.n_trials, callbacks=[_after_trial])

    print("\n=== HPO complete ===")
    print(f"Best trial: #{study.best_trial.number}  val_loss={study.best_value:.6f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(args.output_dir, "all_trials.csv"), index=False)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top3 = sorted(completed, key=lambda t: t.value)[:3]
    top3_configs = [{"trial": t.number, "val_loss": t.value, "params": t.params} for t in top3]
    top3_path = os.path.join(args.output_dir, "top3_configs.json")
    with open(top3_path, "w") as f:
        json.dump(top3_configs, f, indent=2)
    print(f"\nTop-3 configs written to {top3_path}")
    for cfg in top3_configs:
        print(f"  Trial #{cfg['trial']}  val_loss={cfg['val_loss']:.6f}  {cfg['params']}")

    if args.gcs_study_bucket:
        for fname in ["all_trials.csv", "top3_configs.json"]:
            local = os.path.join(args.output_dir, fname)
            if os.path.exists(local):
                _gcs_upload(local, args.gcs_study_bucket, f"film_hpo/{fname}")
        print(f"Results uploaded to gs://{args.gcs_study_bucket}/film_hpo/")


if __name__ == "__main__":
    main()
