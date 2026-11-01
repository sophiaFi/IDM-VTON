"""
Hyperparameter optimisation for IDM-VTON + MeasurementEncoder + Measurement Adapter (Extension 3).

Differences from train_hpo_film.py:
- Measurements enter through a parallel, gated cross-attention branch with its own K/V per
  attn2 layer (src/measurement_adapter.py), not through FiLM / LoRA on ResNet convs.
- Trainable per trial: MeasurementEncoder and MeasurementAdapter (token projector, per-block
  K/V, gates). The UNet, including conv_in, is frozen (it runs in bf16, which would stall conv_in
  updates at small lr). Gates get their own learning-rate multiplier and no weight decay.
- Min-SNR weighting (--snr_gamma, default 5.0) is applied to the training loss only; the
  validation metric is the unweighted MSE + fit loss so trials stay comparable.
- Standard noise schedule (no rescale_betas_zero_snr), as in train_with_measurement_adapter.py.

Usage (called from vertex_hpo_job_adapter.yaml):
  python train_hpo_measurement_adapter.py \
      --data_dir /tmp/hpo/train \
      --pretrained_model_name_or_path /tmp/yisol-IDM-VTON \
      --pretrained_ip_adapter_path /tmp/ckpt/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin \
      --output_dir /tmp/hpo_output \
      --n_trials 20 --max_proxy_steps 1000 --mixed_precision bf16 \
      --gcs_study_bucket idm-vton-bucket --gcs_study_prefix adapter_hpo/study.db
"""

import argparse
import gc
import itertools
import json
import os

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.training_utils import compute_snr
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
from src.measurement_adapter import (
    MeasurementAdapter,
    MeasurementAdapterContext,
    get_hidden_sizes,
    install_measurement_adapter,
)
from src.measurement_encoder import MeasurementEncoder
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel


# ---------------------------------------------------------------------------
# Losses (identical to train_hpo_film.py)
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
    p.add_argument("--num_tokens", type=int, default=16)
    p.add_argument("--segmenter_model", type=str, default="mattmdjaga/segformer_b2_clothes")
    # Fixed (not searched) settings
    p.add_argument("--gradient_accumulation_steps", type=int, default=1,
                   help="Micro-batches per optimizer step. max_proxy_steps counts optimizer steps.")
    p.add_argument("--snr_gamma", type=float, default=5.0,
                   help="Min-SNR gamma for the training loss. <=0 disables it.")
    p.add_argument("--fit_loss_weight", type=float, default=0.7)
    p.add_argument("--fit_loss_alpha_threshold", type=float, default=0.2)
    p.add_argument("--encoder_hidden_dim", type=int, default=256)
    p.add_argument("--encoder_output_dim", type=int, default=512)
    p.add_argument("--num_timestep_bins", type=int, default=10)
    p.add_argument("--num_measurement_tokens", type=int, default=4)
    p.add_argument("--measurement_token_dim", type=int, default=256)
    p.add_argument("--lr_scheduler", type=str, default="constant", choices=["constant", "cosine"],
                   help="Constant by default: cosine decays to 0 within the short proxy run and biases lr upwards.")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Micro-batch size (effective batch = batch_size * gradient_accumulation_steps).")
    # HPO settings
    p.add_argument("--n_trials",          type=int,   default=20)
    p.add_argument("--max_proxy_steps",   type=int,   default=1000,
                   help="Training steps per trial (proxy budget).")
    p.add_argument("--val_batches",       type=int,   default=20,
                   help="Validation batches used to compute the trial metric.")
    p.add_argument("--report_batches",    type=int,   default=10,
                   help="Validation batches for the intermediate (pruner) reports.")
    p.add_argument("--pruner_min_resource", type=int, default=200)
    p.add_argument("--meas_gap", action=argparse.BooleanOptionalAction, default=True,
                   help="At the end of each completed trial also score the val set with measurements "
                        "rolled by one batch and log the difference as user attr `meas_gap` "
                        "(diagnostic only; ~doubles the final evaluation).")
    p.add_argument("--seed",              type=int,   default=0)
    p.add_argument("--val_split",         type=float, default=0.1)
    # GCS study persistence
    p.add_argument("--gcs_study_bucket", type=str, default=None)
    p.add_argument("--gcs_study_prefix", type=str, default="adapter_hpo/study.db")
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

        # Standard schedule, as in train_with_measurement_adapter.py (zero-terminal-SNR breaks Min-SNR).
        self.noise_scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler")
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

def _build_trial_unet(frozen: FrozenComponents):
    """Reconstruct TryonNet from the frozen base state dict (all parameters frozen).

    The Resampler is rebuilt explicitly because UNet2DConditionModel.from_config does not
    include encoder_hid_proj in its architecture, so load_state_dict(strict=False) would
    skip those weights. The IP-Adapter attn processors are created by from_config and
    loaded from the base state dict (processors are part of the UNet state dict).

    Gradient checkpointing is intentionally disabled for HPO (as in train_hpo_film.py):
    the measurement tokens come from a trainable encoder and are read by every checkpointed
    attn2 layer. HPO trials are short (<=1000 steps, batch size 1-2), so the memory is not needed.
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

    # Fully frozen. conv_in is deliberately not trained here: the UNet is cast to bf16 below, and
    # bf16 weights/updates would make conv_in barely move at small lr (full training keeps fp32
    # weights under autocast). The yisol checkpoint already contains the trained 13-channel conv_in.
    unet.requires_grad_(False)
    return unet.to(frozen.device, dtype=frozen.weight_dtype)


# ---------------------------------------------------------------------------
# Batch preparation + loss (shared by training and evaluation)
# ---------------------------------------------------------------------------

def _encode_text(frozen: FrozenComponents, prompts, device):
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


@torch.no_grad()
def _prepare_batch(frozen: FrozenComponents, unet, batch, args, generator=None):
    """Everything that does not need gradients: latents, text/IP conditioning, garment features.

    `generator` makes the VAE sampling, noise and timesteps reproducible (used for validation).
    """
    device = frozen.device
    vae, wdt = frozen.vae, frozen.weight_dtype

    def _lat(x):
        return (vae.encode(x.to(dtype=vae.dtype, device=device)).latent_dist.sample(generator)
                * vae.config.scaling_factor).to(dtype=wdt)

    model_input = _lat(batch["person_image"])
    masked_latents = _lat(batch["masked_person"].reshape(batch["person_image"].shape))
    pose_map = _lat(batch["pose"])

    mask = F.interpolate(batch["mask"].to(device), size=(args.height // 8, args.width // 8)).reshape(
        -1, 1, args.height // 8, args.width // 8
    ).to(dtype=wdt)

    bsz = model_input.shape[0]
    noise = torch.randn(model_input.shape, generator=generator, device=device, dtype=model_input.dtype)
    timesteps = torch.randint(
        0, frozen.noise_scheduler.config.num_train_timesteps, (bsz,), device=device, generator=generator
    )
    noisy_latents = frozen.noise_scheduler.add_noise(model_input, noise, timesteps)
    latent_model_input = torch.cat([noisy_latents, mask, masked_latents, pose_map], dim=1)

    encoder_hidden_states, pooled_text_embeds = _encode_text(frozen, batch["text_prompts"], device)

    add_time_ids = torch.cat([
        torch.tensor([[args.height, args.width, 0, 0, args.height, args.width]],
                     device=device, dtype=wdt)
        for _ in range(bsz)
    ])

    image_embeds = torch.cat(
        [batch["garment_image_clip"][i] for i in range(bsz)], dim=0
    ).to(device, dtype=wdt)
    image_embeds = frozen.image_encoder(image_embeds, output_hidden_states=True).hidden_states[-2]
    ip_tokens = unet.encoder_hid_proj(image_embeds)

    unet_added = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids, "image_embeds": ip_tokens}

    cloth_values = _lat(batch["garment_image"])
    text_embeds_cloth, _ = _encode_text(frozen, batch["text_prompts_cloth"], device)
    _, reference_features = frozen.unet_encoder(
        cloth_values, timesteps, text_embeds_cloth, return_dict=False
    )

    return {
        "latent_model_input": latent_model_input,
        "model_input": model_input,
        "noise": noise,
        "noisy_latents": noisy_latents,
        "timesteps": timesteps,
        "encoder_hidden_states": encoder_hidden_states,
        "unet_added": unet_added,
        "reference_features": list(reference_features),
    }


def _compute_loss(frozen: FrozenComponents, noise_pred, prep, batch, fit_loss_w,
                  fit_alpha_threshold, snr_gamma):
    sched = frozen.noise_scheduler
    timesteps = prep["timesteps"]
    ptype = sched.config.prediction_type
    if ptype == "epsilon":
        target = prep["noise"]
    elif ptype == "v_prediction":
        target = sched.get_velocity(prep["model_input"], prep["noise"], timesteps)
    else:
        target = prep["model_input"]

    if snr_gamma is None:
        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
    else:
        snr = compute_snr(sched, timesteps)
        if ptype == "v_prediction":
            snr = snr + 1
        weights = (
            torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
            / snr.clamp(min=1e-8)
        )
        loss = (F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                .mean(dim=list(range(1, noise_pred.ndim))) * weights).mean()

    if fit_loss_w > 0:
        fl = _compute_fit_loss(
            noise_pred.float(), prep["noisy_latents"].float(), timesteps,
            batch["garment_mask"].to(frozen.device).float(),
            sched, ptype, frozen.vae, frozen.segmenter,
            alpha_threshold=fit_alpha_threshold, boundary_weight=0.5,
        )
        loss = loss + fit_loss_w * fl
    return loss


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def make_objective(frozen: FrozenComponents, train_ds, val_ds, args):
    snr_gamma = args.snr_gamma if args.snr_gamma and args.snr_gamma > 0 else None

    def objective(trial: optuna.Trial) -> float:
        lr                  = trial.suggest_float("lr", 1e-5, 2e-4, log=True)
        gate_lr_mult        = trial.suggest_categorical("gate_lr_multiplier", [1.0, 10.0, 30.0])
        scale_mode          = trial.suggest_categorical("scale_mode", ["global", "block", "timestep"])
        use_fourier         = trial.suggest_categorical("use_fourier", [False, True])
        fourier_scale       = (trial.suggest_float("fourier_scale", 0.1, 1.0, log=True)
                               if use_fourier else 0.25)
        menc_dropout        = trial.suggest_float("measurement_dropout", 0.0, 0.2)
        # Fixed (not searched) to keep the space small for ~30 trials. These are defaults, not
        # tuned optima: tokens/token_dim should be compared in a follow-up run at the best lr.
        num_meas_tokens     = args.num_measurement_tokens
        token_dim           = args.measurement_token_dim
        lr_scheduler_type   = args.lr_scheduler
        batch_size          = args.batch_size
        fit_loss_w         = args.fit_loss_weight
        fit_alpha_threshold = args.fit_loss_alpha_threshold
        #fit_alpha_threshold = trial.suggest_categorical("fit_loss_alpha_threshold", [0.2, 0.3, 0.5])

        torch.manual_seed(args.seed + trial.number)
        device = frozen.device

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=2
        )

        unet = _build_trial_unet(frozen)
        measurement_encoder = MeasurementEncoder(
            num_measurements=9, hidden_dim=args.encoder_hidden_dim,
            output_dim=args.encoder_output_dim, dropout=0.1,
            use_fourier=use_fourier, fourier_scale=fourier_scale,
        ).to(device)
        adapter = MeasurementAdapter(
            input_dim=args.encoder_output_dim,
            hidden_sizes=get_hidden_sizes(unet),
            num_tokens=num_meas_tokens,
            token_dim=token_dim,
            scale_mode=scale_mode,
            num_timestep_bins=args.num_timestep_bins,
        ).to(device)

        adapter_ctx = MeasurementAdapterContext(frozen.noise_scheduler.config.num_train_timesteps)
        hook = install_measurement_adapter(unet, adapter, adapter_ctx)

        def embed_to_tokens(measurements, dropout):
            emb = measurement_encoder(measurements, measurement_dropout_prob=dropout).squeeze(1)
            return adapter(emb)

        adapter_params = [p for n, p in adapter.named_parameters() if n != "gate_table"]
        trainable_params = list(itertools.chain(
            measurement_encoder.parameters(), adapter_params, [adapter.gate_table],
        ))
        optimizer = torch.optim.AdamW(
            [
                {"params": list(itertools.chain(measurement_encoder.parameters(), adapter_params))},
                {"params": [adapter.gate_table], "lr": lr * gate_lr_mult, "weight_decay": 0.0},
            ],
            lr=lr, weight_decay=1e-2,
        )

        if lr_scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_proxy_steps)
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

        try:
            unet.train()
            measurement_encoder.train()
            adapter.train()

            global_step = 0
            train_iter = iter(train_loader)
            report_interval = max(1, args.pruner_min_resource // 2)

            accum = max(1, args.gradient_accumulation_steps)

            # global_step counts optimizer steps, so max_proxy_steps, the pruner and the LR
            # schedule are unaffected by accumulation (each step costs `accum` micro-batches).
            while global_step < args.max_proxy_steps:
                optimizer.zero_grad()
                for _ in range(accum):
                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        batch = next(train_iter)

                    prep = _prepare_batch(frozen, unet, batch, args)

                    # Tokens come from the trainable encoder/adapter (grad-tracked) and must be
                    # cleared only after backward.
                    measurements = batch["measurements"].to(device, dtype=torch.float32)
                    adapter_ctx.tokens = embed_to_tokens(measurements, menc_dropout)

                    noise_pred = unet(
                        prep["latent_model_input"], prep["timesteps"], prep["encoder_hidden_states"],
                        added_cond_kwargs=prep["unet_added"],
                        garment_features=prep["reference_features"],
                    ).sample

                    loss = _compute_loss(frozen, noise_pred, prep, batch, fit_loss_w,
                                         fit_alpha_threshold, snr_gamma)
                    if not torch.isfinite(loss):
                        raise optuna.TrialPruned()

                    (loss / accum).backward()
                    adapter_ctx.tokens = None
                    adapter_ctx.active_tokens = None
                    adapter_ctx.gates = None

                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                global_step += 1

                if global_step % report_interval == 0:
                    val_loss, _ = _evaluate(
                        unet, measurement_encoder, adapter, adapter_ctx, embed_to_tokens,
                        frozen, val_loader, args, fit_loss_w, n_batches=args.report_batches,
                        fit_alpha_threshold=fit_alpha_threshold,
                    )
                    trial.report(val_loss, step=global_step)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            val_loss, val_loss_shifted = _evaluate(
                unet, measurement_encoder, adapter, adapter_ctx, embed_to_tokens,
                frozen, val_loader, args, fit_loss_w, n_batches=args.val_batches,
                fit_alpha_threshold=fit_alpha_threshold, with_shifted=args.meas_gap,
            )
            trial.set_user_attr("final_gates_mean", float(adapter.gate_table.detach().mean()))
            if val_loss_shifted is not None:
                # Diagnostic only (not the objective): > 0 means the model uses the measurements.
                trial.set_user_attr("val_loss_shifted", val_loss_shifted)
                trial.set_user_attr("meas_gap", val_loss_shifted - val_loss)
                print(f"Trial {trial.number}: val_loss={val_loss:.5f}  shifted={val_loss_shifted:.5f}  "
                      f"meas_gap={val_loss_shifted - val_loss:+.5f}")
            return val_loss
        finally:
            adapter_ctx.tokens = None
            adapter_ctx.active_tokens = None
            adapter_ctx.gates = None
            hook.remove()
            del unet, measurement_encoder, adapter, adapter_ctx, optimizer, scheduler
            gc.collect()
            torch.cuda.empty_cache()

    return objective


# ---------------------------------------------------------------------------
# Evaluation (unweighted MSE + fit loss, measurements without dropout)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate(unet, measurement_encoder, adapter, adapter_ctx: MeasurementAdapterContext,
              embed_to_tokens, frozen: FrozenComponents, val_loader, args, fit_loss_w: float,
              n_batches: int, fit_alpha_threshold: float = 0.3, with_shifted: bool = False):
    """Returns (val_loss, shifted_val_loss). shifted_val_loss is None unless `with_shifted`.

    With `with_shifted`, every batch is also scored with the measurements of the previous
    batch (rolled by one, wrapping around), on identical latents, noise and timesteps. The
    difference to val_loss shows whether the model uses the measurements. A roll across
    batches (not a shuffle within one) is used because the HPO runs with batch_size 1.
    """
    unet.eval()
    measurement_encoder.eval()
    adapter.eval()
    device = frozen.device

    # Same VAE samples, noise and timesteps for every trial and every call, so differences in
    # val loss reflect the model and not the noise draw (val_loader is unshuffled).
    gen = torch.Generator(device=device).manual_seed(args.seed + 12345)

    batches = []
    for batch in val_loader:
        batches.append(batch)
        if len(batches) >= n_batches:
            break
    n = len(batches)
    if with_shifted and n < 2:
        with_shifted = False

    def _forward_loss(prep, batch, meas):
        adapter_ctx.tokens = embed_to_tokens(meas.to(device, dtype=torch.float32), 0.0)
        try:
            noise_pred = unet(
                prep["latent_model_input"], prep["timesteps"], prep["encoder_hidden_states"],
                added_cond_kwargs=prep["unet_added"],
                garment_features=prep["reference_features"],
            ).sample
        finally:
            adapter_ctx.tokens = None
            adapter_ctx.active_tokens = None
            adapter_ctx.gates = None
        return _compute_loss(frozen, noise_pred, prep, batch, fit_loss_w,
                             fit_alpha_threshold, snr_gamma=None).item()

    total, total_shifted = 0.0, 0.0
    for i, batch in enumerate(batches):
        prep = _prepare_batch(frozen, unet, batch, args, generator=gen)
        total += _forward_loss(prep, batch, batch["measurements"])
        if with_shifted:
            prev_meas = batches[(i - 1) % n]["measurements"]
            if prev_meas.shape[0] != batch["measurements"].shape[0]:
                # last (smaller) batch of the loader: pad/trim to this batch's size
                prev_meas = prev_meas[: batch["measurements"].shape[0]]
            total_shifted += _forward_loss(prep, batch, prev_meas)

    unet.train()
    measurement_encoder.train()
    adapter.train()
    val_loss = total / max(n, 1)
    return val_loss, (total_shifted / max(n, 1) if with_shifted else None)


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
        study_name="idm_vton_adapter_hpo",
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
    # --n_trials is the study total: after a SPOT preemption + resubmit, only the remainder runs.
    # Trials left RUNNING by the preemption are ignored (neither finished nor counted).
    n_done = sum(t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
                 for t in study.trials)
    n_remaining = args.n_trials - n_done
    print(f"Study has {n_done} finished trials; running {max(n_remaining, 0)} more (target {args.n_trials}).")
    if n_remaining > 0:
        study.optimize(objective, n_trials=n_remaining, callbacks=[_after_trial])

    print("\n=== HPO complete ===")
    print(f"Best trial: #{study.best_trial.number}  val_loss={study.best_value:.6f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(args.output_dir, "all_trials.csv"), index=False)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top3 = sorted(completed, key=lambda t: t.value)[:3]
    top3_configs = [{"trial": t.number, "val_loss": t.value, "meas_gap": t.user_attrs.get("meas_gap"),
                     "params": t.params} for t in top3]
    top3_path = os.path.join(args.output_dir, "top3_configs.json")
    with open(top3_path, "w") as f:
        json.dump(top3_configs, f, indent=2)
    print(f"\nTop-3 configs written to {top3_path}")
    for cfg in top3_configs:
        print(f"  Trial #{cfg['trial']}  val_loss={cfg['val_loss']:.6f}  meas_gap={cfg['meas_gap']}  {cfg['params']}")

    if args.gcs_study_bucket:
        for fname in ["all_trials.csv", "top3_configs.json"]:
            local = os.path.join(args.output_dir, fname)
            if os.path.exists(local):
                _gcs_upload(local, args.gcs_study_bucket, f"adapter_hpo/{fname}")
        print(f"Results uploaded to gs://{args.gcs_study_bucket}/adapter_hpo/")


if __name__ == "__main__":
    main()
