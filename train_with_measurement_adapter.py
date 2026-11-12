"""
Training script for IDM-VTON + Measurement Adapter (Extension 3) on the FIT dataset.

Measurement Adapter (IP-Adapter principle applied to measurements):
- MeasurementEncoder (MLP, optional Fourier features) -> embedding [B, D]
- MeasurementAdapter: embedding -> K tokens (default 4) + per-block trainable K/V projections
- In every TryonNet cross-attention layer (attn2, 70 in SDXL) the output becomes
      Attn(text) + s_ip * Attn(garment IP tokens) + g(block, t) * Attn(measurement tokens)
  The measurement tokens are NOT appended to the context sequence (unlike Ext 1).
- g is trainable: --scale_mode global | block | timestep (see src/measurement_adapter.py).
- to_v of the adapter is zero-initialised, so training starts exactly at the pretrained baseline.

Trainable: MeasurementEncoder, MeasurementAdapter (token projector, K/V, gates), conv_in.
Frozen:    VAE, text/image encoders, GarmentNet, garment IP-Adapter + Resampler, rest of TryonNet.

The measurement tokens travel through a side channel (MeasurementAdapterContext). ctx.tokens must
stay set until after accelerator.backward() because gradient checkpointing recomputes the forward.

Checkpoint layout per saved epoch:
  checkpoint-<step>/
    unet_lora.pt             <- trainable UNet params (conv_in)
    measurement_adapter.pt   <- MeasurementAdapter state dict
    measurement_encoder.pt   <- MeasurementEncoder state dict (includes the Fourier projection buffer)
    adapter_config.json      <- hyper-parameters needed to rebuild the modules at inference
    optimizer.pt
    lr_scheduler.pt
"""

import os
import json
import time
import itertools
import math
from collections import deque
from typing import List

import numpy as np
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
from src.measurement_adapter import (
    MeasurementAdapter, MeasurementAdapterContext, SCALE_MODES,
    get_hidden_sizes, install_measurement_adapter,
)

from ip_adapter.ip_adapter import Resampler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.training_utils import compute_snr

# Shared helpers (identical across the extension scripts)
from train_with_film import (
    _setup_cloud_logging, _mem_snapshot,
    _upload_dir_to_gcs, _download_dir_from_gcs,
    _compute_fit_loss, _SEG_MEAN as _SEG_MEAN_V, _SEG_STD as _SEG_STD_V,
)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="IDM-VTON + Measurement Adapter (Extension 3) training on FIT dataset."
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
    parser.add_argument("--train_batch_size",            type=int,   default=6)
    parser.add_argument("--test_batch_size",             type=int,   default=4)
    parser.add_argument("--val_split",                   type=float, default=0.1)
    parser.add_argument("--num_train_epochs",            type=int,   default=130)
    parser.add_argument("--max_train_steps",             type=int,   default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int,   default=1)
    parser.add_argument("--learning_rate",               type=float, default=1e-4)
    parser.add_argument("--adam_beta1",                  type=float, default=0.9)
    parser.add_argument("--adam_beta2",                  type=float, default=0.999)
    parser.add_argument("--adam_weight_decay",           type=float, default=1e-2)
    parser.add_argument("--adam_epsilon",                type=float, default=1e-8)
    parser.add_argument("--lr_scheduler_type",           type=str,   default="constant",
                        choices=["constant", "cosine"])
    parser.add_argument("--snr_gamma",                   type=float, default=None)
    parser.add_argument("--measurement_dropout",         type=float, default=0.1)
    parser.add_argument("--fit_loss_weight",             type=float, default=0.3)
    parser.add_argument("--fit_loss_boundary_weight",    type=float, default=0.5)
    parser.add_argument("--fit_loss_alpha_threshold",    type=float, default=0.3)
    parser.add_argument("--segmenter_model",             type=str,
                        default="mattmdjaga/segformer_b2_clothes")
    # Anchor: keep the measurement branch from changing the garment appearance.
    parser.add_argument("--anchor_weight", type=float, default=0.0,
                        help="Weight of MSE(pred_branch_on, stopgrad(pred_branch_off)) inside the garment region. "
                             "0 disables the anchor (and its extra no-grad UNet forward).")
    parser.add_argument("--anchor_region", type=str, default="interior", choices=["garment", "interior", "all"],
                        help="garment: GT garment mask; interior: garment mask eroded by --anchor_erode_px "
                             "latent pixels (leaves the silhouette free to move); all: whole latent.")
    parser.add_argument("--anchor_erode_px", type=int, default=3,
                        help="Erosion radius in latent pixels (8 image px each) for --anchor_region interior.")
    # Measurement encoder / adapter
    parser.add_argument("--encoder_hidden_dim", type=int, default=256)
    parser.add_argument("--encoder_output_dim", type=int, default=512,
                        help="Dimension of the measurement embedding fed to the adapter.")
    parser.add_argument("--use_fourier", action=argparse.BooleanOptionalAction, default=True,
                        help="Fourier feature projection of the z-scored measurements.")
    parser.add_argument("--fourier_scale", type=float, default=0.25,
                        help="Std of the random Fourier frequencies (inputs are z-scores, features use 2*pi*x@B).")
    parser.add_argument("--num_measurement_tokens", type=int, default=4)
    parser.add_argument("--measurement_token_dim",  type=int, default=256)
    parser.add_argument("--scale_mode", type=str, default="block", choices=list(SCALE_MODES),
                        help="Trainable measurement-attention weight: global | block | timestep (per block).")
    parser.add_argument("--num_timestep_bins", type=int, default=10,
                        help="Number of gate knots over the timestep for --scale_mode timestep.")
    parser.add_argument("--gate_lr_multiplier", type=float, default=10.0,
                        help="LR multiplier for the gate parameters (they are excluded from weight decay).")
    # Checkpointing / logging
    parser.add_argument("--checkpointing_epoch", type=int, default=1,
                        help="Save a checkpoint every N epochs (0 disables the epoch-end checkpoint).")
    parser.add_argument("--checkpointing_steps", type=int, default=0,
                        help="Also save a checkpoint every N optimizer steps (0 disables).")
    parser.add_argument("--logging_steps",        type=int, default=1000)
    # Validation (generates samples with the measurement branch on and off)
    parser.add_argument("--val_every_steps", type=int, default=None,
                        help="Run validation every N optimizer steps (default: --logging_steps).")
    parser.add_argument("--val_num_batches", type=int, default=1,
                        help="Number of validation batches (of --test_batch_size) used for the metrics.")
    parser.add_argument("--val_metrics", action=argparse.BooleanOptionalAction, default=True,
                        help="Log garment IoU / garment LPIPS for scale 1 vs scale 0 at each validation.")
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


def _grad_norm(params) -> float:
    sq = sum(p.grad.detach().float().norm().item() ** 2 for p in params if p.grad is not None)
    return sq ** 0.5


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
        accelerator.init_trackers("idm-vton-measurement-adapter")

    logger = _setup_cloud_logging("idm-vton-adapter-training") if accelerator.is_main_process else None

    _alloc_history: deque = deque(maxlen=20)

    def _log_mem(tag: str, step: int) -> None:
        if not accelerator.is_main_process:
            return
        snap = _mem_snapshot()
        if not snap:
            return
        logger.info(f"[mem] {tag}  step={step}" + "".join(f"  {k}={v}" for k, v in snap.items()))
        _alloc_history.append(snap["gpu_alloc_gb"])
        if len(_alloc_history) == _alloc_history.maxlen and all(
            _alloc_history[i] < _alloc_history[i + 1] for i in range(len(_alloc_history) - 1)
        ):
            logger.warning(
                f"[mem-leak?] GPU allocated memory has risen monotonically for "
                f"{_alloc_history.maxlen} consecutive samples: "
                f"{_alloc_history[0]:.3f} -> {_alloc_history[-1]:.3f} GB"
            )

    # ── Load models ───────────────────────────────────────────────────────────
    # Standard schedule (no rescale_betas_zero_snr): the pretrained model was trained with it and
    # min-SNR weighting is well defined for every timestep (see fix in commit 0a1da13).
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
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

    # ── Garment IP-Adapter weights ────────────────────────────────────────────
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

    # ── Expand TryonNet input conv 9 -> 13 channels if needed ────────────────
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

    # ── MeasurementEncoder + MeasurementAdapter ──────────────────────────────
    measurement_encoder = MeasurementEncoder(
        num_measurements=9,
        hidden_dim=args.encoder_hidden_dim,
        output_dim=args.encoder_output_dim,
        dropout=0.1,
        use_fourier=args.use_fourier,
        fourier_scale=args.fourier_scale,
    )
    hidden_sizes = get_hidden_sizes(unet)
    measurement_adapter = MeasurementAdapter(
        input_dim=args.encoder_output_dim,
        hidden_sizes=hidden_sizes,
        num_tokens=args.num_measurement_tokens,
        token_dim=args.measurement_token_dim,
        scale_mode=args.scale_mode,
        num_timestep_bins=args.num_timestep_bins,
    )
    adapter_config = {
        "encoder_hidden_dim": args.encoder_hidden_dim,
        "encoder_output_dim": args.encoder_output_dim,
        "use_fourier": args.use_fourier,
        "fourier_scale": args.fourier_scale,
        "num_measurement_tokens": args.num_measurement_tokens,
        "measurement_token_dim": args.measurement_token_dim,
        "scale_mode": args.scale_mode,
        "num_timestep_bins": args.num_timestep_bins,
    }
    if accelerator.is_main_process:
        print(f"Measurement adapter: {len(hidden_sizes)} cross-attention blocks, config={adapter_config}")

    # ── dtype / device placement ──────────────────────────────────────────────
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device)
    # Text encoders stay on CPU and visit the GPU only for the encoding window of each step.
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
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet_encoder.requires_grad_(False)
    image_proj_model.requires_grad_(False)
    unet.requires_grad_(False)          # includes the garment IP-Adapter K/V
    unet.conv_in.requires_grad_(True)   # densepose channels (same as the other extensions)
    measurement_encoder.requires_grad_(True)
    measurement_adapter.requires_grad_(True)

    # ── Garment segmenter for fit loss ────────────────────────────────────────
    segmenter = None
    if args.fit_loss_weight > 0 or args.val_metrics:
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
    gate_params = [measurement_adapter.gate_table]
    adapter_params = [p for n, p in measurement_adapter.named_parameters() if n != "gate_table"]
    optimizer = optimizer_class(
        [
            {"params": list(itertools.chain(trainable_unet_params,
                                            measurement_encoder.parameters(),
                                            adapter_params))},
            {"params": gate_params, "lr": args.learning_rate * args.gate_lr_multiplier, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if accelerator.is_main_process:
        n_unet = sum(p.numel() for p in trainable_unet_params)
        n_menc = sum(p.numel() for p in measurement_encoder.parameters())
        n_adpt = sum(p.numel() for p in measurement_adapter.parameters())
        print(
            f"Trainable params — UNet conv_in: {n_unet:,}  MeasurementEncoder: {n_menc:,}  "
            f"MeasurementAdapter: {n_adpt:,}  total: {n_unet + n_menc + n_adpt:,}"
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
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_train_steps)
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    # ── Accelerator prepare ───────────────────────────────────────────────────
    (
        unet, measurement_encoder, measurement_adapter,
        image_proj_model, unet_encoder, image_encoder,
        optimizer, train_dataloader, test_dataloader,
    ) = accelerator.prepare(
        unet, measurement_encoder, measurement_adapter,
        image_proj_model, unet_encoder, image_encoder,
        optimizer, train_dataloader, test_dataloader,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    _log_mem("after-accelerator-prepare", 0)

    # ── Install the measurement branch (after prepare: modules are on the right device) ──
    adapter_ctx    = MeasurementAdapterContext(noise_scheduler.config.num_train_timesteps)
    unet_raw       = accelerator.unwrap_model(unet)
    adapter_raw    = accelerator.unwrap_model(measurement_adapter)
    menc_raw       = accelerator.unwrap_model(measurement_encoder)
    _adapter_hook  = install_measurement_adapter(unet_raw, adapter_raw, adapter_ctx)

    def _embed_to_tokens(measurements: torch.Tensor, dropout: float) -> torch.Tensor:
        emb = menc_raw(measurements, measurement_dropout_prob=dropout).squeeze(1)  # [B, D]
        return adapter_raw(emb)                                                    # [B, K, token_dim]

    # ── Resume from checkpoint ────────────────────────────────────────────────
    first_epoch = 0
    global_step = 0

    if args.resume_from_checkpoint is not None:
        resume_path = args.resume_from_checkpoint
        ckpt_name = os.path.basename(resume_path.rstrip("/\\"))
        if ckpt_name.startswith("checkpoint-"):
            global_step = int(ckpt_name.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

        if resume_path.startswith("gs://"):
            without_scheme = resume_path[len("gs://"):]
            bucket_name, gcs_prefix = without_scheme.split("/", 1)
            local_resume = os.path.join(args.output_dir, "resumed_checkpoint")
            if accelerator.is_main_process:
                _download_dir_from_gcs(bucket_name, gcs_prefix, local_resume)
            accelerator.wait_for_everyone()
            resume_path = local_resume

        if accelerator.is_main_process:
            print(f"Resuming from {resume_path} (global_step={global_step}, first_epoch={first_epoch})")

        unet_raw.load_state_dict(
            torch.load(os.path.join(resume_path, "unet_lora.pt"), map_location="cpu"), strict=False
        )
        adapter_raw.load_state_dict(
            torch.load(os.path.join(resume_path, "measurement_adapter.pt"), map_location="cpu")
        )
        menc_raw.load_state_dict(
            torch.load(os.path.join(resume_path, "measurement_encoder.pt"), map_location="cpu")
        )

        opt_path   = os.path.join(resume_path, "optimizer.pt")
        sched_path = os.path.join(resume_path, "lr_scheduler.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
        if os.path.exists(sched_path):
            lr_scheduler.load_state_dict(torch.load(sched_path, map_location="cpu"))

    # ── Validation metrics (mirror evaluate.py: SegFormer ids {4,5,7}, torchmetrics VGG LPIPS) ──
    _val_lpips = {}

    @torch.no_grad()
    def _val_sample_metrics(gen_pil, target_m11: torch.Tensor, gt_garment_mask: torch.Tensor) -> dict:
        """gen_pil: generated PIL image; target_m11: [3,H,W] in [-1,1]; gt_garment_mask: [1,H,W] {0,1}."""
        dev = accelerator.device
        gen = torch.from_numpy(np.asarray(gen_pil.convert("RGB"))).permute(2, 0, 1).float().div(255).to(dev)
        tgt = ((target_m11.float() + 1.0) / 2.0).clamp(0, 1).to(dev)
        gt_mask = gt_garment_mask.float().to(dev)
        H, W = gen.shape[-2:]

        def seg_mask(img01: torch.Tensor) -> torch.Tensor:
            x = (img01.unsqueeze(0) - _SEG_MEAN_V.to(dev)) / _SEG_STD_V.to(dev)
            x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
            logits = segmenter(pixel_values=x).logits.float()
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
            pred = logits.argmax(dim=1)
            return sum((pred == lid).float() for lid in (4, 5, 7)).clamp(0, 1)  # [1,H,W]

        def iou(a, b):
            return ((a * b).sum() / ((a + b).clamp(0, 1).sum() + 1e-8)).item()

        gen_seg, tgt_seg = seg_mask(gen), seg_mask(tgt)
        out = {
            # Symmetric (generated vs target, both through SegFormer) = evaluate.py's garment_iou
            "garment_iou":      iou(gen_seg, tgt_seg),
            # Generated vs the annotated GT silhouette = what the fit loss optimises
            "garment_iou_gt":   iou(gen_seg, gt_mask),
        }
        if "m" not in _val_lpips and not _val_lpips.get("failed"):
            try:
                from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
                _val_lpips["m"] = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(dev)
            except Exception as e:  # typically: `lpips` package missing in the image
                _val_lpips["failed"] = True
                logger.warning(f"[val-metrics] LPIPS unavailable, logging IoU only: {e!r}")
        if "m" in _val_lpips:
            lp = _val_lpips["m"]
            out["lpips"] = lp((gen * 2 - 1).unsqueeze(0), (tgt * 2 - 1).unsqueeze(0)).item()
            # Garment-region LPIPS: everything outside the GT garment silhouette zeroed in both images
            out["garment_lpips"] = lp(
                (gen * gt_mask * 2 - 1).unsqueeze(0), (tgt * gt_mask * 2 - 1).unsqueeze(0)
            ).item()
        return out

    # ── Validation helper ─────────────────────────────────────────────────────
    def _run_validation(tag: str) -> None:
        if not accelerator.is_main_process:
            return

        text_encoder.to(accelerator.device)
        text_encoder_2.to(accelerator.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            newpipe = TryonPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=unet_raw,
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

            metric_rows = {"scale1": [], "scale0": []}

            for batch_idx, sample in enumerate(test_dataloader):
                if batch_idx >= max(1, args.val_num_batches):
                    break
                meas = sample["measurements"].to(accelerator.device, dtype=torch.float32)

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
                    dev = accelerator.device

                    def _generate(branch_on: bool):
                        # branch off == ctx.tokens None == measurement_scale 0 at inference.
                        # The pre-hook on the UNet repeats the tokens for classifier-free guidance.
                        adapter_ctx.tokens = _embed_to_tokens(meas, 0.0) if branch_on else None
                        generator = torch.Generator(newpipe.device).manual_seed(args.seed) if args.seed else None
                        try:
                            return newpipe(
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
                        finally:
                            adapter_ctx.tokens = None

                    images = _generate(True)
                    torch.cuda.empty_cache()  # release the fp32 VAE-decode buffers before the second pass
                    images_off = _generate(False) if args.val_metrics else None
                    torch.cuda.empty_cache()

                for i, img in enumerate(images):
                    if batch_idx == 0:
                        img.save(os.path.join(args.output_dir, f"{tag}_{i}_test.jpg"))
                        if images_off is not None:
                            images_off[i].save(os.path.join(args.output_dir, f"{tag}_{i}_scale0.jpg"))
                        person_np = (
                            (sample["person_image"][i].cpu().float() + 1.0) / 2.0
                        ).clamp(0, 1).permute(1, 2, 0).mul(255).byte().numpy()
                        from PIL import Image as _PILImage
                        _PILImage.fromarray(person_np).save(
                            os.path.join(args.output_dir, f"{tag}_{i}_input.jpg")
                        )

                if args.val_metrics and images_off is not None:
                    try:
                        for key, imgs in (("scale1", images), ("scale0", images_off)):
                            for i, img in enumerate(imgs):
                                metric_rows[key].append(_val_sample_metrics(
                                    img, sample["person_image"][i], sample["garment_mask"][i],
                                ))
                    except Exception as e:  # metrics must never kill training
                        logger.warning(f"[val-metrics] skipped for batch {batch_idx}: {e!r}")

            if args.val_metrics and metric_rows["scale1"] and metric_rows["scale0"]:
                payload = {}
                for key, rows in metric_rows.items():
                    for name in rows[0]:
                        payload[f"val/{name}_{key}"] = float(np.mean([r[name] for r in rows]))
                for name in metric_rows["scale1"][0]:
                    payload[f"val/{name}_delta"] = payload[f"val/{name}_scale1"] - payload[f"val/{name}_scale0"]
                accelerator.log(payload, step=global_step)
                logger.info(f"[val-metrics] tag={tag} step={global_step} n={len(metric_rows['scale1'])} "
                            + "  ".join(f"{k}={v:.4f}" for k, v in payload.items()))

            del newpipe
            torch.cuda.empty_cache()

        text_encoder.to('cpu')
        text_encoder_2.to('cpu')
        torch.cuda.empty_cache()

    def _save_checkpoint(step: int) -> None:
        if not accelerator.is_main_process:
            return
        save_path = os.path.join(args.output_dir, f"checkpoint-{step}")
        os.makedirs(save_path, exist_ok=True)

        torch.save(
            {k: v.data.cpu() for k, v in unet_raw.named_parameters() if v.requires_grad},
            os.path.join(save_path, "unet_lora.pt"),
        )
        torch.save(adapter_raw.state_dict(), os.path.join(save_path, "measurement_adapter.pt"))
        torch.save(menc_raw.state_dict(),    os.path.join(save_path, "measurement_encoder.pt"))
        with open(os.path.join(save_path, "adapter_config.json"), "w") as f:
            json.dump(adapter_config, f, indent=2)
        torch.save(optimizer.state_dict(),    os.path.join(save_path, "optimizer.pt"))
        torch.save(lr_scheduler.state_dict(), os.path.join(save_path, "lr_scheduler.pt"))

        if args.gcs_checkpoint_bucket:
            ckpt_folder = os.path.basename(save_path)
            gcs_prefix  = f"{args.gcs_checkpoint_prefix}/{ckpt_folder}"
            _upload_dir_to_gcs(save_path, args.gcs_checkpoint_bucket, gcs_prefix)

    # ── Training loop ─────────────────────────────────────────────────────────
    profiling = args.profile_steps > 0
    profile_step_times = []

    if profiling and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    max_steps = args.profile_steps if profiling else args.max_train_steps
    progress_bar = tqdm(
        range(max_steps), desc="Steps",
        initial=global_step,
        disable=not accelerator.is_local_main_process,
    )
    train_loss     = 0.0
    train_fit_loss = 0.0
    train_anchor_loss = 0.0
    last_val_step = -1
    last_ckpt_step = global_step if args.resume_from_checkpoint else -1
    adapter_grad_norm_accum = 0.0
    menc_grad_norm_accum    = 0.0

    unet.train()
    measurement_encoder.train()
    measurement_adapter.train()

    for epoch in range(first_epoch, args.num_train_epochs):
        for _, batch in enumerate(train_dataloader):
            step_start = time.perf_counter()
            with accelerator.accumulate(unet):

                # ── Validation samples ────────────────────────────────────────
                if (global_step > 0 and global_step != last_val_step
                        and global_step % (args.val_every_steps or args.logging_steps) == 0):
                    last_val_step = global_step  # global_step stays constant across accumulation micro-batches
                    unet.eval()
                    measurement_encoder.eval()
                    measurement_adapter.eval()
                    _run_validation(tag=str(global_step))
                    unet.train()
                    measurement_encoder.train()
                    measurement_adapter.train()

                # ── Encode images to latents ──────────────────────────────────
                pixel_values   = batch["person_image"].to(dtype=vae.dtype)
                model_input    = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                masked_latents = vae.encode(
                    batch["masked_person"].reshape(batch["person_image"].shape).to(dtype=vae.dtype)
                ).latent_dist.sample() * vae.config.scaling_factor

                masks = batch["mask"]
                mask  = torch.nn.functional.interpolate(
                    masks, size=(args.height // 8, args.width // 8)
                ).reshape(-1, 1, args.height // 8, args.width // 8)

                pose_map = vae.encode(batch["pose"].to(dtype=vae.dtype)).latent_dist.sample() * vae.config.scaling_factor

                noise     = torch.randn_like(model_input)
                bsz       = model_input.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,),
                    device=model_input.device
                )
                noisy_latents      = noise_scheduler.add_noise(model_input, noise, timesteps)
                latent_model_input = torch.cat([noisy_latents, mask, masked_latents, pose_map], dim=1)

                # ── Text encoders: GPU only for the encoding window ───────────
                text_encoder.to(accelerator.device)
                text_encoder_2.to(accelerator.device)

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
                # Context stays [B, 77, 2048]: measurements are NOT appended (own attention branch).
                encoder_hidden_states = torch.cat([text_embeds, text_embeds_2], dim=-1)

                # ── Measurement tokens -> adapter context ────────────────────
                measurements = batch["measurements"].to(accelerator.device, dtype=torch.float32)
                adapter_ctx.tokens = _embed_to_tokens(measurements, args.measurement_dropout)
                # NOT cleared before backward: gradient checkpointing recomputes the forward.

                # ── SDXL time/size conditioning ───────────────────────────────
                def compute_time_ids(original_size, crops_coords_top_left=(0, 0)):
                    add_time_ids = list(original_size + crops_coords_top_left + (args.height, args.width))
                    return torch.tensor([add_time_ids]).to(accelerator.device)

                add_time_ids = torch.cat([compute_time_ids((args.height, args.width)) for _ in range(bsz)])

                # ── Garment IP-Adapter: CLIP image embeddings ────────────────
                image_embeds = torch.cat([batch["garment_image_clip"][i] for i in range(bsz)], dim=0)
                image_embeds = image_encoder(image_embeds, output_hidden_states=True).hidden_states[-2]
                ip_tokens    = image_proj_model(image_embeds)

                unet_added_cond_kwargs = {
                    "text_embeds":  pooled_text_embeds,
                    "time_ids":     add_time_ids,
                    "image_embeds": ip_tokens,
                }

                # ── GarmentNet -> reference features ─────────────────────────
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

                text_encoder.to('cpu')
                text_encoder_2.to('cpu')
                torch.cuda.empty_cache()

                _, reference_features = unet_encoder(cloth_values, timesteps, text_embeds_cloth, return_dict=False)
                reference_features = list(reference_features)

                # ── Anchor target: same forward with the measurement branch off (no grad) ──
                # conv_in is shared, so this isolates the effect of the measurement branch only.
                # The pre-hook recomputes active_tokens/gates on every forward, so restoring
                # ctx.tokens afterwards is enough for the main (checkpointed) forward.
                noise_pred_off = None
                if args.anchor_weight > 0:
                    meas_tokens = adapter_ctx.tokens
                    adapter_ctx.tokens = None
                    with torch.no_grad():
                        noise_pred_off = unet(
                            latent_model_input, timesteps, encoder_hidden_states,
                            added_cond_kwargs=unet_added_cond_kwargs,
                            garment_features=reference_features,
                        ).sample
                    adapter_ctx.tokens = meas_tokens

                # ── TryonNet forward (measurement branch active in every attn2) ──
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

                if noise_pred_off is not None:
                    if args.anchor_region == "all":
                        region = torch.ones_like(mask)
                    else:
                        region = F.interpolate(
                            batch["garment_mask"].to(accelerator.device).float(),
                            size=mask.shape[-2:], mode="nearest",
                        )
                        if args.anchor_region == "interior" and args.anchor_erode_px > 0:
                            k = 2 * args.anchor_erode_px + 1
                            region = -F.max_pool2d(-region, kernel_size=k, stride=1, padding=k // 2)
                    sq_err = (noise_pred.float() - noise_pred_off.float().detach()).pow(2)
                    denom = (region.sum() * sq_err.shape[1]).clamp(min=1.0)
                    anchor_loss = (sq_err * region).sum() / denom
                    loss = loss + args.anchor_weight * anchor_loss

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                if args.fit_loss_weight > 0:
                    avg_fit_loss = accelerator.gather(fit_loss.repeat(args.train_batch_size)).mean()
                    train_fit_loss += avg_fit_loss.item() / args.gradient_accumulation_steps
                if noise_pred_off is not None:
                    avg_anchor = accelerator.gather(anchor_loss.detach().repeat(args.train_batch_size)).mean()
                    train_anchor_loss += avg_anchor.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)

                # Cleared only after backward (gradient checkpointing recompute).
                adapter_ctx.tokens        = None
                adapter_ctx.active_tokens = None
                adapter_ctx.gates         = None

                menc_grad_norm_accum    += _grad_norm(measurement_encoder.parameters()) / args.gradient_accumulation_steps
                adapter_grad_norm_accum += _grad_norm(measurement_adapter.parameters()) / args.gradient_accumulation_steps

                if accelerator.sync_gradients:
                    all_trainable = (
                        [p for p in unet.parameters() if p.requires_grad]
                        + list(measurement_encoder.parameters())
                        + list(measurement_adapter.parameters())
                    )
                    accelerator.clip_grad_norm_(all_trainable, 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    progress_bar.update(1)
                    global_step += 1

                    if (args.checkpointing_steps > 0 and global_step % args.checkpointing_steps == 0
                            and global_step != last_ckpt_step):
                        _save_checkpoint(global_step)
                        last_ckpt_step = global_step

                    if profiling:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        profile_step_times.append(time.perf_counter() - step_start)

                    snap = _mem_snapshot() if accelerator.is_main_process else {}
                    gt = adapter_raw.gate_table.detach().float()
                    log_payload = {
                        "train_loss":           train_loss,
                        "menc_grad_norm":       menc_grad_norm_accum,
                        "adapter_grad_norm":    adapter_grad_norm_accum,
                        "gate_mean":            gt.mean().item(),
                        "gate_min":             gt.min().item(),
                        "gate_max":             gt.max().item(),
                        "adapter_v_weight_rms": (
                            sum(v.weight.detach().float().pow(2).mean() for v in adapter_raw.to_v)
                            / len(adapter_raw.to_v)
                        ).sqrt().item(),
                    }
                    if args.fit_loss_weight > 0:
                        log_payload["train_fit_loss"] = train_fit_loss
                    if args.anchor_weight > 0:
                        log_payload["train_anchor_loss"] = train_anchor_loss
                    if snap:
                        log_payload.update({k: v for k, v in snap.items() if "peak" not in k})
                    accelerator.log(log_payload, step=global_step)
                    train_loss     = 0.0
                    train_fit_loss = 0.0
                    train_anchor_loss = 0.0
                    menc_grad_norm_accum    = 0.0
                    adapter_grad_norm_accum = 0.0

            if args.memory_log_steps > 0 and global_step % args.memory_log_steps == 0:
                _log_mem("train-step", global_step)

            progress_bar.set_postfix(step_loss=loss.detach().item())

            if global_step >= max_steps:
                break

        # ── Epoch memory summary ──────────────────────────────────────────────
        if accelerator.is_main_process and torch.cuda.is_available():
            snap = _mem_snapshot()
            logger.info(f"[mem] epoch={epoch} end  " + "  ".join(f"{k}={v}" for k, v in snap.items()))
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
                "adapter_blocks":              len(hidden_sizes),
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
                report["gpu_peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
                report["gpu_peak_reserved_gb"]  = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
                props = torch.cuda.get_device_properties(0)
                report["gpu_name"]            = props.name
                report["gpu_total_memory_gb"] = round(props.total_memory / 1024**3, 2)

            report_path = os.path.join(args.output_dir, "resource_profile.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nResource profile written to {report_path}")
            print(json.dumps(report, indent=2))
            break

        # ── Checkpoint ────────────────────────────────────────────────────────
        epoch_ckpt_due = args.checkpointing_epoch > 0 and epoch % args.checkpointing_epoch == 0
        if epoch_ckpt_due and global_step != last_ckpt_step:
            _save_checkpoint(global_step)
            last_ckpt_step = global_step

    # ── Final validation ──────────────────────────────────────────────────────
    unet.eval()
    measurement_encoder.eval()
    measurement_adapter.eval()
    _run_validation(tag="final")
    accelerator.end_training()
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
