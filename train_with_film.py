"""
Training script for IDM-VTON + MeasurementEncoder + FiLM Extension on the FIT dataset.

FiLM (Feature-wise Linear Modulation) extension:
- Measurements condition TryonNet DIRECTLY via element-wise feature modulation,
  bypassing the attention mechanism used in the base (token-concatenation) extension.
- FiLM(h | m) = γ(m) ⊙ h + β(m), injected after norm2 and before the SiLU nonlinearity
  in the target ResNet blocks.
- Target blocks (lowest spatial resolution): down_blocks[-1] (DownBlock2D, deepest),
  mid_block (UNetMidBlock2DCrossAttn), up_blocks[0] (UpBlock2D, first decoder block).
- FiLM Generator Heads are initialized with γ ≈ 1, β ≈ 0 so training begins near
  the pretrained baseline and measurement influence grows gradually.

Parameter-efficient fine-tuning strategy (LoRA on ResNet convs):
- MeasurementEncoder: trained in full.
- FiLMGeneratorSet (one head per target ResNet block): trained in full.
- Deep ResNet blocks (down_blocks[-1], mid_block.resnets, up_blocks[0]):
    conv1 and conv2 wrapped with LoRAConv2d (rank --lora_rank, default 4).
    Also trains: time_emb_proj, norm1, norm2 (small layers; ~57M total vs ~251M without LoRA).
    Attention layers in mid_block remain frozen.
- conv_in (9→13 ch expansion for densepose, same as base extension): trained in full.
- VAE, text encoders, image encoder, GarmentNet, IP-Adapter / Resampler,
  attention layers in deep blocks, all outer TryonNet blocks: frozen.
- Text encoders stay on CPU; moved to GPU only for the brief encoding window per step.

FiLM hooks are registered via PyTorch forward hooks on each target resnet.norm2 module.
A shared _FiLMContext object passes the current batch's measurement embedding into the
hooks without changing the UNet's public API.  IMPORTANT: film_ctx.embedding must remain
set throughout accelerator.backward() because gradient checkpointing recomputes the
forward pass during the backward; clearing it before backward would give wrong gradients.

Checkpoint layout per saved epoch:
  checkpoint-<step>/
    unet_lora.pt           ← trainable UNet params only (LoRA A/B + norms + time_emb_proj + conv_in)
    film_generators.pt     ← FiLMGeneratorSet state dict
    measurement_encoder.pt ← MeasurementEncoder state dict
    optimizer.pt
    lr_scheduler.pt
"""

import os
import json
import time
import itertools
import math
import logging as py_logging
from collections import deque
from typing import List, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import (
    CLIPTextModel, CLIPTokenizer,
    CLIPVisionModelWithProjection, CLIPTextModelWithProjection,
    SegformerForSemanticSegmentation,
)
from tqdm.auto import tqdm

from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.fit_dataset import FITDatasetWithMeasurements
from src.measurement_encoder import MeasurementEncoder
from src.film_generator import FiLMGeneratorSet

from ip_adapter.ip_adapter import Resampler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.training_utils import compute_snr


# ---------------------------------------------------------------------------
# FiLM hook infrastructure
# ---------------------------------------------------------------------------

class _FiLMContext:
    """Mutable context object shared between the training loop and FiLM hooks.

    The training loop sets `embedding` before each UNet forward pass and clears
    it after the corresponding backward pass (not before, because gradient
    checkpointing recomputes the forward during the backward).
    """
    embedding: Optional[torch.Tensor] = None


class LoRAConv2d(torch.nn.Module):
    """Frozen Conv2d with a trainable low-rank bypass ΔW = B·A.

    Approximates the full weight update with two small convolutions:
      lora_A: (C_in → rank, 1×1) down-projects the spatial feature map
      lora_B: (rank → C_out, k×k) up-projects back to the original spatial size

    lora_B is zero-initialized so ΔW = 0 at the start of training, preserving
    the pretrained baseline exactly.
    """

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
    """Replace conv1, conv2 (and conv_shortcut if present) with LoRA-wrapped versions."""
    for resnet in resnets:
        resnet.conv1 = LoRAConv2d(resnet.conv1, rank=rank)
        resnet.conv2 = LoRAConv2d(resnet.conv2, rank=rank)
        if getattr(resnet, "conv_shortcut", None) is not None:
            resnet.conv_shortcut = LoRAConv2d(resnet.conv_shortcut, rank=rank)


def _get_target_resnets(unet) -> list:
    """Return the ResNet blocks targeted for FiLM injection.

    These are the blocks at the lowest spatial resolution:
      - deepest DownBlock (down_blocks[-1])
      - MidBlock resnets
      - first UpBlock (up_blocks[0])
    """
    resnets = []
    resnets.extend(list(unet.down_blocks[-1].resnets))
    resnets.extend(list(unet.mid_block.resnets))
    resnets.extend(list(unet.up_blocks[0].resnets))
    return resnets


def _register_film_hooks(unet, film_generators: FiLMGeneratorSet, ctx: _FiLMContext) -> list:
    """Register output hooks on norm2 of each target ResNet block.

    The hook fires right after GroupNorm (norm2) computes its output and before
    the SiLU nonlinearity, applying:
        FiLM(h | m) = γ(m) * h + β(m)

    During classifier-free guidance the UNet receives a doubled batch
    [uncond; cond]; the embedding is repeated to match automatically.

    Returns a list of hook handles — call handle.remove() to deregister.
    """
    target_resnets = _get_target_resnets(unet)
    if len(target_resnets) != len(film_generators.heads):
        raise RuntimeError(
            f"FiLM head count mismatch: {len(target_resnets)} target ResNet blocks "
            f"but {len(film_generators.heads)} FiLM heads. "
            "Ensure FiLMGeneratorSet was built from the same UNet."
        )

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
                    # CFG doubles the batch; repeat embedding to match
                    emb = emb.repeat(B_out // B_emb, 1)
                gamma, beta = h(emb.to(device=out.device, dtype=out.dtype))
                return gamma[:, :, None, None] * out + beta[:, :, None, None]
            return _hook
        hooks.append(resnet.norm2.register_forward_hook(_make_hook(head)))

    return hooks


# ---------------------------------------------------------------------------
# Utilities (identical to train_with_measurements.py)
# ---------------------------------------------------------------------------

def _setup_cloud_logging(log_name: str = "idm-vton-film-training") -> py_logging.Logger:
    logger = py_logging.getLogger(log_name)
    logger.setLevel(py_logging.DEBUG)
    if not logger.handlers:
        try:
            import google.cloud.logging as gcl
            gcl.Client().setup_logging()
            logger.info("Cloud Logging handler attached.")
        except Exception:
            handler = py_logging.StreamHandler()
            handler.setFormatter(py_logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(handler)
    return logger


def _mem_snapshot(device=None) -> dict:
    if not torch.cuda.is_available():
        return {}
    d = device or torch.cuda.current_device()
    return {
        "gpu_alloc_gb":        round(torch.cuda.memory_allocated(d) / 1024**3, 3),
        "gpu_reserved_gb":     round(torch.cuda.memory_reserved(d) / 1024**3, 3),
        "gpu_peak_alloc_gb":   round(torch.cuda.max_memory_allocated(d) / 1024**3, 3),
        "gpu_peak_reserved_gb":round(torch.cuda.max_memory_reserved(d) / 1024**3, 3),
    }


def _upload_dir_to_gcs(local_dir: str, bucket_name: str, prefix: str) -> None:
    try:
        from google.cloud import storage as gcs
    except ImportError:
        print("google-cloud-storage not installed — skipping GCS upload.")
        return
    client = gcs.Client()
    bucket = client.bucket(bucket_name)
    local_dir = os.path.abspath(local_dir)
    for dirpath, _, filenames in os.walk(local_dir):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            rel = os.path.relpath(fpath, local_dir)
            blob_name = f"{prefix}/{rel}".replace("\\", "/")
            bucket.blob(blob_name).upload_from_filename(fpath)
    print(f"Uploaded {local_dir} → gs://{bucket_name}/{prefix}/")


def _download_dir_from_gcs(bucket_name: str, prefix: str, local_dir: str) -> None:
    try:
        from google.cloud import storage as gcs
    except ImportError:
        raise RuntimeError("google-cloud-storage is required. pip install google-cloud-storage")
    client = gcs.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix + "/"))
    if not blobs:
        raise FileNotFoundError(f"No objects found at gs://{bucket_name}/{prefix}/")
    for blob in blobs:
        rel = blob.name[len(prefix) + 1:]
        if not rel:
            continue
        dest = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        blob.download_to_filename(dest)
    print(f"Downloaded gs://{bucket_name}/{prefix}/ → {local_dir}")


_SEGFORMER_GARMENT_IDS = (4, 7)
_SEG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_SEG_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_SEG_SIZE = 512


def _boundary_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pool   = torch.nn.functional.max_pool2d
    erode  = lambda x: -torch.nn.functional.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
    boundary_pred = pool(pred, kernel_size=3, stride=1, padding=1) - erode(pred)
    boundary_gt   = pool(gt,   kernel_size=3, stride=1, padding=1) - erode(gt)
    n_boundary = (boundary_gt > 0.1).float().sum().clamp(min=1.0)
    return (boundary_pred - boundary_gt).abs().sum() / n_boundary


def _compute_fit_loss(
    noise_pred, noisy_latents, timesteps, gt_garment_mask,
    noise_scheduler, prediction_type, vae, segmenter,
    alpha_threshold=0.3, boundary_weight=0.5,
) -> torch.Tensor:
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(noisy_latents.device)
    alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(noisy_latents.dtype)
    keep = alpha_t.view(-1) > alpha_threshold
    if not keep.any():
        return torch.tensor(0.0, device=noisy_latents.device, requires_grad=False)

    noise_pred    = noise_pred[keep]
    noisy_latents = noisy_latents[keep]
    alpha_t       = alpha_t[keep]
    gt_mask       = gt_garment_mask[keep].clamp(0, 1).to(noisy_latents.dtype)

    if prediction_type == "epsilon":
        sqrt_alpha = alpha_t.sqrt()
        sqrt_one_minus = (1.0 - alpha_t).sqrt()
        x0_pred = (noisy_latents - sqrt_one_minus * noise_pred) / sqrt_alpha.clamp(min=1e-8)
    elif prediction_type == "v_prediction":
        sqrt_alpha = alpha_t.sqrt()
        sqrt_one_minus = (1.0 - alpha_t).sqrt()
        x0_pred = sqrt_alpha * noisy_latents - sqrt_one_minus * noise_pred
    else:
        x0_pred = noise_pred

    decoded = vae.decode(
        (x0_pred / vae.config.scaling_factor).to(dtype=vae.dtype)
    ).sample.float()

    decoded_01 = (decoded.clamp(-1, 1) + 1.0) / 2.0
    seg_mean = _SEG_MEAN.to(decoded.device, dtype=decoded.dtype)
    seg_std  = _SEG_STD.to(decoded.device, dtype=decoded.dtype)
    seg_input = (decoded_01 - seg_mean) / seg_std
    seg_input_rs = F.interpolate(seg_input, size=(_SEG_SIZE, _SEG_SIZE), mode="bilinear", align_corners=False)

    logits = segmenter(
        pixel_values=seg_input_rs.to(dtype=next(segmenter.parameters()).dtype)
    ).logits.float()

    H, W = gt_mask.shape[-2], gt_mask.shape[-1]
    garment_logits = sum(logits[:, cls_id:cls_id+1] for cls_id in _SEGFORMER_GARMENT_IDS)
    soft_pred = torch.sigmoid(garment_logits)
    soft_pred = F.interpolate(soft_pred, size=(H, W), mode="bilinear", align_corners=False)

    intersection = (soft_pred * gt_mask).sum(dim=(1, 2, 3))
    union = (soft_pred + gt_mask - soft_pred * gt_mask).sum(dim=(1, 2, 3)).clamp(min=1e-8)
    iou_loss = (1.0 - intersection / union).mean()
    b_loss = _boundary_loss(soft_pred, gt_mask)

    return iou_loss + boundary_weight * b_loss


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="IDM-VTON + MeasurementEncoder + FiLM training on FIT dataset."
    )
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="yisol/IDM-VTON")
    parser.add_argument("--pretrained_garmentnet_path", type=str, default=None)
    parser.add_argument("--pretrained_ip_adapter_path", type=str, default=None)
    parser.add_argument("--image_encoder_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="gs://ma-idm-vton-data/datasets/fit-mini")
    parser.add_argument("--output_dir", type=str, default="output")
    # Image dimensions
    parser.add_argument("--width",  type=int, default=680)
    parser.add_argument("--height", type=int, default=512)
    # Training
    parser.add_argument("--train_batch_size",           type=int,   default=6)
    parser.add_argument("--test_batch_size",            type=int,   default=4)
    parser.add_argument("--val_split",                  type=float, default=0.1)
    parser.add_argument("--num_train_epochs",           type=int,   default=130)
    parser.add_argument("--max_train_steps",            type=int,   default=None)
    parser.add_argument("--gradient_accumulation_steps",type=int,   default=1)
    parser.add_argument("--learning_rate",              type=float, default=1e-4)
    parser.add_argument("--adam_beta1",                 type=float, default=0.9)
    parser.add_argument("--adam_beta2",                 type=float, default=0.999)
    parser.add_argument("--adam_weight_decay",          type=float, default=1e-2)
    parser.add_argument("--adam_epsilon",               type=float, default=1e-8)
    parser.add_argument("--lr_scheduler_type",          type=str,   default="constant",
                        choices=["constant", "cosine"])
    parser.add_argument("--snr_gamma",                  type=float, default=None)
    parser.add_argument("--noise_offset",               type=float, default=None)
    parser.add_argument("--measurement_dropout",        type=float, default=0.1)
    parser.add_argument("--lora_rank",                  type=int,   default=4,
                        help="LoRA rank for ResNet conv layers (0 = full fine-tuning, higher = more params).")
    parser.add_argument("--fit_loss_weight",            type=float, default=0.3)
    parser.add_argument("--fit_loss_boundary_weight",   type=float, default=0.5)
    parser.add_argument("--fit_loss_alpha_threshold",   type=float, default=0.3)
    parser.add_argument("--segmenter_model",            type=str,
                        default="mattmdjaga/segformer_b2_clothes")
    # Checkpointing / logging
    parser.add_argument("--checkpointing_epoch", type=int, default=1)
    parser.add_argument("--logging_steps",        type=int, default=1000)
    # Efficiency
    parser.add_argument("--mixed_precision",   type=str,  default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--use_8bit_adam",     action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    # IP-Adapter
    parser.add_argument("--num_tokens", type=int, default=16)
    # Inference (validation samples)
    parser.add_argument("--guidance_scale",      type=float, default=2.0)
    parser.add_argument("--num_inference_steps", type=int,   default=30)
    parser.add_argument("--seed",                type=int,   default=42)
    # GCS checkpoint persistence
    parser.add_argument("--gcs_checkpoint_bucket", type=str, default=None)
    parser.add_argument("--gcs_checkpoint_prefix", type=str, default="checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    # Misc
    parser.add_argument("--local_rank",       type=int, default=-1)
    parser.add_argument("--profile_steps",    type=int, default=0)
    parser.add_argument("--memory_log_steps", type=int, default=10)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=ProjectConfiguration(project_dir=args.output_dir),
        log_with="tensorboard",
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        accelerator.init_trackers("idm-vton-film")

    logger = _setup_cloud_logging() if accelerator.is_main_process else None

    _alloc_history: deque = deque(maxlen=20)

    def _log_mem(tag: str, step: int) -> None:
        if not accelerator.is_main_process:
            return
        snap = _mem_snapshot()
        if not snap:
            return
        msg = f"[mem] {tag}  step={step}" + "".join(f"  {k}={v}" for k, v in snap.items())
        logger.info(msg)
        alloc = snap["gpu_alloc_gb"]
        _alloc_history.append(alloc)
        if len(_alloc_history) == _alloc_history.maxlen and all(
            _alloc_history[i] < _alloc_history[i + 1] for i in range(len(_alloc_history) - 1)
        ):
            logger.warning(
                f"[mem-leak?] GPU allocated memory has risen monotonically for "
                f"{_alloc_history.maxlen} consecutive samples: "
                f"{_alloc_history[0]:.3f} → {_alloc_history[-1]:.3f} GB"
            )

    # ── Load models ───────────────────────────────────────────────────────────
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", rescale_betas_zero_snr=True
    )
    tokenizer   = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder   = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=torch.float16
    )

    image_encoder_path      = args.image_encoder_path or args.pretrained_model_name_or_path
    image_encoder_subfolder = None if args.image_encoder_path else "image_encoder"
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        image_encoder_path, subfolder=image_encoder_subfolder
    )

    garmentnet_path      = args.pretrained_garmentnet_path or args.pretrained_model_name_or_path
    garmentnet_subfolder = "unet" if args.pretrained_garmentnet_path else "unet_encoder"
    unet_encoder = UNet2DConditionModel_ref.from_pretrained(garmentnet_path, subfolder=garmentnet_subfolder)
    unet_encoder.config.addition_embed_type = None
    unet_encoder.config["addition_embed_type"] = None

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet",
        low_cpu_mem_usage=False, device_map=None,
    )
    unet.config.encoder_hid_dim      = image_encoder.config.hidden_size
    unet.config.encoder_hid_dim_type = "ip_image_proj"
    unet.config["encoder_hid_dim"]      = image_encoder.config.hidden_size
    unet.config["encoder_hid_dim_type"] = "ip_image_proj"

    # ── IP-Adapter weights ────────────────────────────────────────────────────
    if args.pretrained_ip_adapter_path:
        ip_bin = args.pretrained_ip_adapter_path
    else:
        ip_bin = os.path.join(args.pretrained_model_name_or_path, "ip_adapter", "ip-adapter-plus_sdxl_vit-h.bin")
    state_dict = torch.load(ip_bin, map_location="cpu")
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

    image_proj_model = Resampler(
        dim=image_encoder.config.hidden_size,
        depth=4, dim_head=64, heads=20,
        num_queries=args.num_tokens,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4,
    ).to(accelerator.device, dtype=torch.float32)
    image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
    unet.encoder_hid_proj = image_proj_model

    # ── Expand TryonNet input conv 9 → 13 channels if needed ─────────────────
    if unet.conv_in.in_channels == 9:
        conv_new = torch.nn.Conv2d(
            in_channels=13, out_channels=unet.conv_in.out_channels, kernel_size=3, padding=1
        )
        torch.nn.init.zeros_(conv_new.weight)
        conv_new.weight.data[:, :9] = unet.conv_in.weight.data
        conv_new.bias.data = unet.conv_in.bias.data
        unet.conv_in = conv_new
        unet.config["in_channels"] = 13
        unet.config.in_channels = 13

    # ── MeasurementEncoder ────────────────────────────────────────────────────
    cross_attn_dim = unet.config.cross_attention_dim  # 2048 for SDXL
    measurement_encoder = MeasurementEncoder(
        num_measurements=9,
        hidden_dim=256,
        output_dim=cross_attn_dim,
        dropout=0.1,
        use_fourier=False,
    )

    # ── FiLM Generator Heads ──────────────────────────────────────────────────
    # One head per target ResNet block; channel_dim comes from resnet.norm2.num_channels
    # (= out_channels of that ResNet block, always 1280 for the deep SDXL blocks).
    target_resnets_raw = _get_target_resnets(unet)
    film_channel_dims  = [r.norm2.num_channels for r in target_resnets_raw]
    film_generators    = FiLMGeneratorSet(input_dim=cross_attn_dim, channel_dims=film_channel_dims)

    # ── LoRA wrapping ─────────────────────────────────────────────────────────
    # Wrap conv1/conv2 (and conv_shortcut when present) in the target ResNet blocks
    # with low-rank adapters.  This reduces ResNet trainable params from ~206M to ~1.4M
    # (rank 4), saving ~2 GB of optimizer state.  lora_B is zero-init so the pretrained
    # baseline is preserved at step 0.
    if args.lora_rank > 0:
        _apply_lora_to_resnets(_get_target_resnets(unet), rank=args.lora_rank)

    if accelerator.is_main_process:
        print(
            f"FiLM targets: {len(target_resnets_raw)} ResNet blocks "
            f"(channels: {film_channel_dims}), LoRA rank={args.lora_rank}"
        )

    # ── dtype / device placement ──────────────────────────────────────────────
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device)
    # Keep text encoders on CPU — they are ~2.3 GB and only needed briefly per step.
    # They are moved to GPU in the training loop just before encoding, then back to CPU.
    text_encoder.to(dtype=weight_dtype)
    text_encoder_2.to(dtype=weight_dtype)
    image_encoder.to(accelerator.device,  dtype=weight_dtype)
    unet_encoder.to(accelerator.device,   dtype=weight_dtype)

    # ── Memory optimizations ──────────────────────────────────────────────────
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available.")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # ── Freeze strategy ───────────────────────────────────────────────────────
    # Frozen: VAE, text/image encoders, GarmentNet, IP-Adapter, outer TryonNet blocks,
    #         attention layers inside the deep mid-block.
    # Trainable: deep ResNet blocks (down[-1], mid resnets, up[0]), conv_in,
    #            MeasurementEncoder, FiLMGeneratorSet.
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet_encoder.requires_grad_(False)
    image_proj_model.requires_grad_(False)

    unet.requires_grad_(False)

    # conv_in: new densepose channels must be learned from scratch
    unet.conv_in.requires_grad_(True)

    if args.lora_rank > 0:
        # LoRA mode: selectively unfreeze LoRA A/B weights + small conditioning layers.
        # The frozen base conv weights inside LoRAConv2d stay frozen.
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
    else:
        # Full fine-tuning fallback (--lora_rank 0)
        unet.down_blocks[-1].requires_grad_(True)
        for resnet in unet.mid_block.resnets:
            resnet.requires_grad_(True)
        unet.up_blocks[0].requires_grad_(True)

    measurement_encoder.requires_grad_(True)
    film_generators.requires_grad_(True)

    # ── Garment segmenter for fit loss ────────────────────────────────────────
    segmenter = None
    if args.fit_loss_weight > 0:
        segmenter = SegformerForSemanticSegmentation.from_pretrained(args.segmenter_model)
        segmenter.requires_grad_(False)
        segmenter.eval()
        segmenter.to(accelerator.device, dtype=torch.float32)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Install bitsandbytes for 8-bit Adam: `pip install bitsandbytes`.")
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    trainable_unet_params = [p for p in unet.parameters() if p.requires_grad]
    params_to_opt = itertools.chain(
        trainable_unet_params,
        measurement_encoder.parameters(),
        film_generators.parameters(),
    )
    optimizer = optimizer_class(
        params_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if accelerator.is_main_process:
        n_unet  = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        n_menc  = sum(p.numel() for p in measurement_encoder.parameters())
        n_film  = sum(p.numel() for p in film_generators.parameters())
        print(
            f"Trainable params — UNet deep blocks+conv_in: {n_unet:,}  "
            f"MeasurementEncoder: {n_menc:,}  FiLMGenerators: {n_film:,}  "
            f"total: {n_unet + n_menc + n_film:,}"
        )
        _log_mem("after-model-load", 0)

    # ── Datasets ──────────────────────────────────────────────────────────────
    full_train_dataset = FITDatasetWithMeasurements(
        data_root=args.data_dir, phase="train", size=(args.height, args.width)
    )
    n_val   = max(1, int(len(full_train_dataset) * args.val_split))
    n_train = len(full_train_dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed if args.seed is not None else 0),
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, pin_memory=True, shuffle=True,
        batch_size=args.train_batch_size, num_workers=4,
    )
    test_dataloader = torch.utils.data.DataLoader(
        val_dataset, shuffle=False, batch_size=args.test_batch_size, num_workers=4,
    )

    # ── Step count ────────────────────────────────────────────────────────────
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    overrode_max_train_steps   = args.max_train_steps is None
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    if args.lr_scheduler_type == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_train_steps
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    # ── Accelerator prepare ───────────────────────────────────────────────────
    (
        unet, measurement_encoder, film_generators,
        image_proj_model, unet_encoder, image_encoder,
        optimizer, train_dataloader, test_dataloader,
    ) = accelerator.prepare(
        unet, measurement_encoder, film_generators,
        image_proj_model, unet_encoder, image_encoder,
        optimizer, train_dataloader, test_dataloader,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    _log_mem("after-accelerator-prepare", 0)

    # ── Register FiLM hooks (after prepare so modules are on the right device) ─
    film_ctx   = _FiLMContext()
    unet_raw   = accelerator.unwrap_model(unet)
    film_gen_raw = accelerator.unwrap_model(film_generators)
    _film_hooks = _register_film_hooks(unet_raw, film_gen_raw, film_ctx)

    if accelerator.is_main_process:
        print(f"Registered {len(_film_hooks)} FiLM hooks on target ResNet norm2 layers.")

    # ── Resume from checkpoint ────────────────────────────────────────────────
    first_epoch = 0
    global_step = 0

    if args.resume_from_checkpoint is not None:
        resume_path = args.resume_from_checkpoint
        if resume_path.startswith("gs://"):
            without_scheme = resume_path[len("gs://"):]
            bucket_name, gcs_prefix = without_scheme.split("/", 1)
            local_resume = os.path.join(args.output_dir, "resumed_checkpoint")
            if accelerator.is_main_process:
                _download_dir_from_gcs(bucket_name, gcs_prefix, local_resume)
            accelerator.wait_for_everyone()
            resume_path = local_resume

        if accelerator.is_main_process:
            print(f"Resuming from {resume_path}")

        unet_unwrapped = accelerator.unwrap_model(unet)

        # Trainable UNet params (LoRA A/B + norms + time_emb_proj + conv_in).
        # LoRA wrappers are already applied above, so the keys exist in the state dict.
        unet_lora_state = torch.load(os.path.join(resume_path, "unet_lora.pt"), map_location="cpu")
        unet_unwrapped.load_state_dict(unet_lora_state, strict=False)

        # FiLM generators
        film_state = torch.load(os.path.join(resume_path, "film_generators.pt"), map_location="cpu")
        accelerator.unwrap_model(film_generators).load_state_dict(film_state)

        # MeasurementEncoder
        menc_state = torch.load(os.path.join(resume_path, "measurement_encoder.pt"), map_location="cpu")
        accelerator.unwrap_model(measurement_encoder).load_state_dict(menc_state)

        opt_path   = os.path.join(resume_path, "optimizer.pt")
        sched_path = os.path.join(resume_path, "lr_scheduler.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
        if os.path.exists(sched_path):
            lr_scheduler.load_state_dict(torch.load(sched_path, map_location="cpu"))

        ckpt_name = os.path.basename(resume_path.rstrip("/\\"))
        if ckpt_name.startswith("checkpoint-"):
            global_step = int(ckpt_name.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    # ── Validation helper ─────────────────────────────────────────────────────
    def _run_validation(tag: str) -> None:
        """Run one validation batch through the full pipeline and save images."""
        if not accelerator.is_main_process:
            return

        menc_unwrapped = accelerator.unwrap_model(measurement_encoder)
        unet_unwrapped = accelerator.unwrap_model(unet)

        # Text encoders are on CPU during training — move them to GPU for validation.
        text_encoder.to(accelerator.device)
        text_encoder_2.to(accelerator.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            # Build a TryonPipeline around the current (unwrapped) UNet.
            # The FiLM hooks are already registered on unet_unwrapped.down_blocks[-1] etc.,
            # so setting film_ctx.embedding before the pipeline call is sufficient.
            newpipe = TryonPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=unet_unwrapped,
                vae=vae,
                scheduler=noise_scheduler,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                image_encoder=image_encoder,
                unet_encoder=unet_encoder,
                feature_extractor=None,
                torch_dtype=torch.float16,
                add_watermarker=False,
                safety_checker=None,
            ).to(accelerator.device)

            for sample in test_dataloader:
                meas = sample["measurements"].to(accelerator.device, dtype=torch.float32)
                # Compute embedding [B, 2048] — no dropout during validation
                meas_emb = menc_unwrapped(meas, measurement_dropout_prob=0.0).squeeze(1)
                # Set context so FiLM hooks activate during inference denoising steps.
                # During CFG the UNet receives 2B inputs; the hook handles batch-size
                # mismatch automatically.
                film_ctx.embedding = meas_emb.to(dtype=torch.float32)

                img_emb_list = [sample["garment_image_clip"][i] for i in range(sample["garment_image_clip"].shape[0])]
                num_prompts  = len(img_emb_list)
                image_embeds = torch.cat(img_emb_list, dim=0).to(accelerator.device)

                prompt          = sample["text_prompts"]
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                if not isinstance(prompt, List):
                    prompt = [prompt] * num_prompts
                if not isinstance(negative_prompt, List):
                    negative_prompt = [negative_prompt] * num_prompts

                with torch.inference_mode():
                    (
                        prompt_embeds, negative_prompt_embeds,
                        pooled_prompt_embeds, negative_pooled_prompt_embeds,
                    ) = newpipe.encode_prompt(
                        prompt, num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )
                    prompt_cloth          = sample["text_prompts_cloth"]
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
                    dev = accelerator.device
                    images = newpipe(
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                        num_inference_steps=args.num_inference_steps,
                        generator=generator,
                        strength=1.0,
                        pose_img=sample["pose"].to(dev, dtype=torch.float16),
                        text_embeds_cloth=prompt_embeds_c,
                        cloth=sample["garment_image"].to(dev, dtype=torch.float16),
                        mask_image=sample["mask"].to(dev, dtype=torch.float16),
                        image=((sample["person_image"] + 1.0) / 2.0).to(dev, dtype=torch.float16),
                        height=args.height,
                        width=args.width,
                        guidance_scale=args.guidance_scale,
                        ip_adapter_image=image_embeds,
                    )[0]

                film_ctx.embedding = None  # safe to clear after full inference

                for i, img in enumerate(images):
                    img.save(os.path.join(args.output_dir, f"{tag}_{i}_test.jpg"))
                    person_np = (
                        (sample["person_image"][i].cpu().float() + 1.0) / 2.0
                    ).clamp(0, 1).permute(1, 2, 0).mul(255).byte().numpy()
                    from PIL import Image as _PILImage
                    _PILImage.fromarray(person_np).save(
                        os.path.join(args.output_dir, f"{tag}_{i}_input.jpg")
                    )
                break

            del newpipe
            torch.cuda.empty_cache()

        # Move text encoders back to CPU after validation.
        text_encoder.to('cpu')
        text_encoder_2.to('cpu')
        torch.cuda.empty_cache()

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
    train_loss      = 0.0
    train_fit_loss  = 0.0
    film_grad_norm_accum = 0.0
    menc_grad_norm_accum = 0.0

    unet.train()
    measurement_encoder.train()
    film_generators.train()

    for epoch in range(first_epoch, args.num_train_epochs):
        for _, batch in enumerate(train_dataloader):
            step_start = time.perf_counter()
            with accelerator.accumulate(unet):

                # ── Validation samples ────────────────────────────────────────
                if global_step % args.logging_steps == 0:
                    unet.eval()
                    measurement_encoder.eval()
                    film_generators.eval()
                    _run_validation(tag=str(global_step))
                    unet.train()
                    measurement_encoder.train()
                    film_generators.train()

                # ── Encode images to latents ──────────────────────────────────
                pixel_values  = batch["person_image"].to(dtype=vae.dtype)
                model_input   = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                masked_latents = vae.encode(
                    batch["masked_person"].reshape(batch["person_image"].shape).to(dtype=vae.dtype)
                ).latent_dist.sample() * vae.config.scaling_factor

                masks = batch["mask"]
                mask  = torch.nn.functional.interpolate(
                    masks, size=(args.height // 8, args.width // 8)
                ).reshape(-1, 1, args.height // 8, args.width // 8)

                pose_map = vae.encode(batch["pose"].to(dtype=vae.dtype)).latent_dist.sample() * vae.config.scaling_factor

                noise       = torch.randn_like(model_input)
                bsz         = model_input.shape[0]
                timesteps   = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,),
                    device=model_input.device
                )
                noisy_latents      = noise_scheduler.add_noise(model_input, noise, timesteps)
                latent_model_input = torch.cat([noisy_latents, mask, masked_latents, pose_map], dim=1)

                # ── Text encoders: GPU only for the encoding window ───────────
                text_encoder.to(accelerator.device)
                text_encoder_2.to(accelerator.device)

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
                text_embeds        = enc1.hidden_states[-2]   # [B, 77, 768]
                pooled_text_embeds = enc2[0]                  # [B, 1280]
                text_embeds_2      = enc2.hidden_states[-2]   # [B, 77, 1280]
                # Cross-attention context: [B, 77, 2048] — no measurement token appended;
                # measurements are injected via FiLM instead.
                encoder_hidden_states = torch.cat([text_embeds, text_embeds_2], dim=-1)

                # ── Measurement embedding → FiLM context ─────────────────────
                measurements    = batch["measurements"].to(accelerator.device, dtype=torch.float32)
                measurement_emb = accelerator.unwrap_model(measurement_encoder)(
                    measurements, measurement_dropout_prob=args.measurement_dropout
                ).squeeze(1)  # [B, 2048]
                # Keep in float32 for numerical stability; hook casts to feature dtype
                film_ctx.embedding = measurement_emb.to(dtype=torch.float32)
                # NOTE: do NOT clear film_ctx.embedding here — it must remain set through
                # accelerator.backward() because gradient checkpointing recomputes the
                # forward pass during the backward.

                # ── SDXL time/size conditioning ───────────────────────────────
                def compute_time_ids(original_size, crops_coords_top_left=(0, 0)):
                    add_time_ids = list(original_size + crops_coords_top_left + (args.height, args.width))
                    return torch.tensor([add_time_ids]).to(accelerator.device)

                add_time_ids = torch.cat([compute_time_ids((args.height, args.width)) for _ in range(bsz)])

                # ── IP-Adapter: garment CLIP image embeddings ─────────────────
                image_embeds = torch.cat([batch["garment_image_clip"][i] for i in range(bsz)], dim=0)
                image_embeds = image_encoder(image_embeds, output_hidden_states=True).hidden_states[-2]
                ip_tokens    = image_proj_model(image_embeds)

                unet_added_cond_kwargs = {
                    "text_embeds": pooled_text_embeds,
                    "time_ids":    add_time_ids,
                    "image_embeds": ip_tokens,
                }

                # ── GarmentNet → reference features ──────────────────────────
                cloth_values = vae.encode(
                    batch["garment_image"].to(accelerator.device, dtype=vae.dtype)
                ).latent_dist.sample() * vae.config.scaling_factor

                text_input_ids_c   = tokenizer(
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

                # Encoders no longer needed — move back to CPU to free ~2.3 GB before UNet forward.
                text_encoder.to('cpu')
                text_encoder_2.to('cpu')
                torch.cuda.empty_cache()

                _, reference_features = unet_encoder(cloth_values, timesteps, text_embeds_cloth, return_dict=False)
                reference_features = list(reference_features)

                # ── TryonNet forward — FiLM hooks fire inside here ────────────
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
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                        / snr.clamp(min=1e-8)
                    )
                    loss = (F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                            .mean(dim=list(range(1, noise_pred.ndim))) * weights).mean()

                if args.fit_loss_weight > 0:
                    fit_loss = _compute_fit_loss(
                        noise_pred=noise_pred.float(),
                        noisy_latents=noisy_latents.float(),
                        timesteps=timesteps,
                        gt_garment_mask=batch["garment_mask"].to(accelerator.device).float(),
                        noise_scheduler=noise_scheduler,
                        prediction_type=noise_scheduler.config.prediction_type,
                        vae=vae,
                        segmenter=segmenter,
                        alpha_threshold=args.fit_loss_alpha_threshold,
                        boundary_weight=args.fit_loss_boundary_weight,
                    )
                    loss = loss + args.fit_loss_weight * fit_loss

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                if args.fit_loss_weight > 0:
                    avg_fit_loss = accelerator.gather(fit_loss.repeat(args.train_batch_size)).mean()
                    train_fit_loss += avg_fit_loss.item() / args.gradient_accumulation_steps

                # Backward: gradient checkpointing may recompute forward; film_ctx.embedding
                # must still be set so the FiLM hooks produce the same outputs.
                accelerator.backward(loss)

                # film_ctx cleared AFTER backward is complete
                film_ctx.embedding = None

                # Accumulate grad norms for logging
                menc_grad_sq = sum(
                    p.grad.detach().norm().item() ** 2
                    for p in measurement_encoder.parameters() if p.grad is not None
                )
                menc_grad_norm_accum += menc_grad_sq ** 0.5 / args.gradient_accumulation_steps

                film_grad_sq = sum(
                    p.grad.detach().norm().item() ** 2
                    for p in film_generators.parameters() if p.grad is not None
                )
                film_grad_norm_accum += film_grad_sq ** 0.5 / args.gradient_accumulation_steps

                if accelerator.sync_gradients:
                    all_trainable = (
                        [p for p in unet.parameters() if p.requires_grad]
                        + list(measurement_encoder.parameters())
                        + list(film_generators.parameters())
                    )
                    accelerator.clip_grad_norm_(all_trainable, 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    progress_bar.update(1)
                    global_step += 1
                    if profiling:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        profile_step_times.append(time.perf_counter() - step_start)

                    snap = _mem_snapshot() if accelerator.is_main_process else {}
                    log_payload = {
                        "train_loss":        train_loss,
                        "menc_grad_norm":    menc_grad_norm_accum,
                        "film_grad_norm":    film_grad_norm_accum,
                    }
                    if args.fit_loss_weight > 0:
                        log_payload["train_fit_loss"] = train_fit_loss
                    if snap:
                        log_payload.update({k: v for k, v in snap.items() if "peak" not in k})
                    accelerator.log(log_payload, step=global_step)
                    train_loss           = 0.0
                    train_fit_loss       = 0.0
                    menc_grad_norm_accum = 0.0
                    film_grad_norm_accum = 0.0

            if args.memory_log_steps > 0 and global_step % args.memory_log_steps == 0:
                _log_mem("train-step", global_step)

            progress_bar.set_postfix(step_loss=loss.detach().item())

            if global_step >= max_steps:
                break

        # ── Epoch memory summary ──────────────────────────────────────────────
        if accelerator.is_main_process and torch.cuda.is_available():
            snap = _mem_snapshot()
            logger.info(
                f"[mem] epoch={epoch} end  "
                + "  ".join(f"{k}={v}" for k, v in snap.items())
            )
            torch.cuda.reset_peak_memory_stats()

        # ── Profile report ────────────────────────────────────────────────────
        if profiling and global_step >= max_steps and accelerator.is_main_process:
            report = {
                "profile_steps":               args.profile_steps,
                "train_batch_size":            args.train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_batch_size":        args.train_batch_size * args.gradient_accumulation_steps,
                "mixed_precision":             args.mixed_precision,
                "gradient_checkpointing":      args.gradient_checkpointing,
                "enable_xformers":             args.enable_xformers_memory_efficient_attention,
                "film_heads":                  len(film_generators),
            }
            if profile_step_times:
                times = profile_step_times[1:]
                if times:
                    avg_s = sum(times) / len(times)
                    report["step_time_avg_s"]          = round(avg_s, 3)
                    report["step_time_min_s"]          = round(min(times), 3)
                    report["step_time_max_s"]          = round(max(times), 3)
                    report["throughput_samples_per_s"] = round(args.train_batch_size / avg_s, 2)
            if torch.cuda.is_available():
                peak_bytes    = torch.cuda.max_memory_allocated()
                reserved_bytes = torch.cuda.max_memory_reserved()
                report["gpu_peak_allocated_gb"] = round(peak_bytes     / 1024**3, 2)
                report["gpu_peak_reserved_gb"]  = round(reserved_bytes / 1024**3, 2)
                props = torch.cuda.get_device_properties(0)
                report["gpu_name"]             = props.name
                report["gpu_total_memory_gb"]  = round(props.total_memory / 1024**3, 2)

            report_path = os.path.join(args.output_dir, "resource_profile.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nResource profile written to {report_path}")
            print(json.dumps(report, indent=2))
            break

        # ── Checkpoint ────────────────────────────────────────────────────────
        if epoch % args.checkpointing_epoch == 0 and accelerator.is_main_process:
            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            os.makedirs(save_path, exist_ok=True)

            unet_unwrapped = accelerator.unwrap_model(unet)

            # Save only trainable UNet params (LoRA A/B + norms + time_emb_proj + conv_in).
            # Frozen base weights are not saved — they come from the pretrained checkpoint.
            torch.save(
                {k: v.data.cpu() for k, v in unet_unwrapped.named_parameters() if v.requires_grad},
                os.path.join(save_path, "unet_lora.pt"),
            )

            # FiLM generators
            torch.save(
                accelerator.unwrap_model(film_generators).state_dict(),
                os.path.join(save_path, "film_generators.pt"),
            )

            # MeasurementEncoder
            torch.save(
                accelerator.unwrap_model(measurement_encoder).state_dict(),
                os.path.join(save_path, "measurement_encoder.pt"),
            )

            # Optimizer and LR scheduler
            torch.save(optimizer.state_dict(),    os.path.join(save_path, "optimizer.pt"))
            torch.save(lr_scheduler.state_dict(), os.path.join(save_path, "lr_scheduler.pt"))

            if args.gcs_checkpoint_bucket:
                ckpt_folder = os.path.basename(save_path)
                gcs_prefix  = f"{args.gcs_checkpoint_prefix}/{ckpt_folder}"
                _upload_dir_to_gcs(save_path, args.gcs_checkpoint_bucket, gcs_prefix)

    # ── Final validation ──────────────────────────────────────────────────────
    unet.eval()
    measurement_encoder.eval()
    film_generators.eval()
    _run_validation(tag="final")
    accelerator.end_training()
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
