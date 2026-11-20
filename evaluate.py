"""
evaluate.py — Evaluation pipeline for IDM-VTON (with optional measurement conditioning).

Usage — compare a result folder against ground truth:

    python evaluate.py \\
        --result_dir result_fit \\
        --data_dir data/data_mini_test \\
        --output_dir eval_output

    # With measurement counterfactual analysis. First generate the +/- sigma images with any of the
    # three inference scripts (inference_fit.py / inference_film.py / inference_measurement_adapter.py):
    python inference_film.py --data_dir data/data_mini_test --checkpoint_dir output/checkpoint-500 \\
        --output_dir result_film --num_samples 8 --counterfactual --cf_sigma 2.0
    # then analyse them (reads result_film/cf/<dimension>/<stem>_{plus,minus}.png):
    python evaluate.py \\
        --result_dir result_film \\
        --data_dir data/data_mini_test \\
        --cf_dir result_film/cf \\
        --output_dir eval_output

Computed metrics (following docs/evaluation.md):

  Image quality (vs. ground truth):
    SSIM, PSNR, LPIPS

  Distribution quality (generated set vs. ground truth set):
    FID, KID

  Garment region fidelity (inside predicted mask):
    garment_ssim, garment_lpips

  Garment mask quality (segmented mask vs. GT mask):
    garment_iou, garment_boundary_f1

  Semantic similarity:
    clip_sim   — CLIP ViT-B/32 cosine similarity of generated vs. target
    dino_sim   — DINO ViT-S/8 cosine similarity outside garment region

  Measurement controllability (--cf_dir):
    For each base measurement, the inference scripts generate ±σ variants with everything else
    (inputs, seed, steps) fixed; the derived ease features are recomputed consistently. This script
    segments the garment in each pair and reports the |area| change plus the signed change in area,
    bbox height and bbox width.

All per-sample results are written to <output_dir>/results.csv.
A summary JSON is written to <output_dir>/summary.json.
Side-by-side comparison images are written to <output_dir>/visuals/.
"""

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torchmetrics.image.kid import KernelInceptionDistance as KID
from transformers import (
    CLIPModel,
    CLIPProcessor,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# peft==0.6.0 LoraConfig doesn't accept kwargs added in later versions
# (e.g. loftq_config).  Patch __init__ to drop unknown keyword arguments.
try:
    from peft import LoraConfig as _LoraConfig
    _lora_known = set(inspect.signature(_LoraConfig.__init__).parameters)
    _lora_orig_init = _LoraConfig.__init__

    def _lora_permissive_init(self, *args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k in _lora_known}
        _lora_orig_init(self, *args, **kwargs)

    _LoraConfig.__init__ = _lora_permissive_init
except Exception:
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_image_rgb(path: str, size: Optional[tuple] = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.LANCZOS)
    return img


def pil_to_tensor_01(img: Image.Image) -> torch.Tensor:
    """HWC uint8 PIL → CHW float32 [0, 1]."""
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def pil_to_tensor_norm(img: Image.Image) -> torch.Tensor:
    """HWC uint8 PIL → CHW float32 [-1, 1]."""
    return pil_to_tensor_01(img) * 2.0 - 1.0


def mask_to_binary(mask_pil: Image.Image, threshold: float = 0.5) -> torch.Tensor:
    """Grayscale PIL mask → [1, H, W] binary float32."""
    arr = np.array(mask_pil.convert("L"), dtype=np.float32) / 255.0
    binary = (arr > threshold).astype(np.float32)
    return torch.from_numpy(binary).unsqueeze(0)


def apply_mask_outside(img_t: torch.Tensor, mask_t: torch.Tensor) -> torch.Tensor:
    """Zero-out pixels inside the mask (keep background). img: [C,H,W], mask: [1,H,W]."""
    return img_t * (1.0 - mask_t)


def apply_mask_inside(img_t: torch.Tensor, mask_t: torch.Tensor) -> torch.Tensor:
    """Zero-out pixels outside the mask (keep garment region). img: [C,H,W], mask: [1,H,W]."""
    return img_t * mask_t


def ssim_numpy(img1: np.ndarray, img2: np.ndarray) -> float:
    """SSIM via skimage; input: HWC uint8."""
    return structural_similarity(img1, img2, channel_axis=2, data_range=255)


def psnr_numpy(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def iou_binary(mask_pred: torch.Tensor, mask_gt: torch.Tensor) -> float:
    """IoU of two binary tensors."""
    inter = (mask_pred * mask_gt).sum().item()
    union = ((mask_pred + mask_gt).clamp(0, 1)).sum().item()
    return inter / (union + 1e-8)


def _boundary_maps(
    mask_pred: torch.Tensor, mask_gt: torch.Tensor, dilation: int = 3
):
    """Return (p_boundary, g_boundary, dilated_p_boundary, dilated_g_boundary).

    All tensors are [1,1,H,W] float32.

    Boundary = 1-px contour on the mask edge, extracted as mask minus its erosion.
    Dilated boundary = boundary fattened by `dilation` px (tolerance band for F1 hits
    and for visualising near-misses).
    """
    erode_kernel = torch.ones(1, 1, 3, 3, device=mask_pred.device)
    dil_kernel   = torch.ones(1, 1, dilation * 2 + 1, dilation * 2 + 1, device=mask_pred.device)

    p = mask_pred.float().view(1, 1, mask_pred.shape[-2], mask_pred.shape[-1])
    g = mask_gt.float().view(1, 1, mask_gt.shape[-2], mask_gt.shape[-1])

    # 1-px interior boundary: mask minus erosion
    p_eroded = -F.max_pool2d(-p, kernel_size=3, stride=1, padding=1)
    g_eroded = -F.max_pool2d(-g, kernel_size=3, stride=1, padding=1)
    p_boundary = (p - p_eroded).clamp(0, 1)
    g_boundary = (g - g_eroded).clamp(0, 1)

    # Dilate each boundary for hit-tolerance
    dilated_p_boundary = (F.conv2d(p_boundary, dil_kernel, padding=dilation) > 0).float()
    dilated_g_boundary = (F.conv2d(g_boundary, dil_kernel, padding=dilation) > 0).float()

    return p_boundary, g_boundary, dilated_p_boundary, dilated_g_boundary


def boundary_f1(mask_pred: torch.Tensor, mask_gt: torch.Tensor, dilation: int = 3) -> float:
    """Boundary F1: fraction of boundary pixels within `dilation` px of the opposing boundary."""
    p_boundary, g_boundary, dilated_p_boundary, dilated_g_boundary = _boundary_maps(
        mask_pred, mask_gt, dilation
    )
    # precision: how many pred boundary pixels are near the GT boundary?
    p_hits = (p_boundary * dilated_g_boundary).sum().item()
    # recall: how many GT boundary pixels are near the pred boundary?
    g_hits = (g_boundary * dilated_p_boundary).sum().item()
    precision = p_hits / (p_boundary.sum().item() + 1e-8)
    recall    = g_hits / (g_boundary.sum().item() + 1e-8)
    return 2 * precision * recall / (precision + recall + 1e-8)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a.flatten().unsqueeze(0), dim=1)
    b = F.normalize(b.flatten().unsqueeze(0), dim=1)
    return (a * b).sum().item()


def mask_overlap_image(
    base_img: Image.Image,
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    size: tuple,
) -> Image.Image:
    """Render TP/FP/FN mask overlap on top of base_img at display size.

    Green = true positive, Red = false positive, Blue = false negative.
    """
    img = base_img.resize(size, Image.LANCZOS)
    base = np.array(img, dtype=np.float32)
    W, H = size

    def _resize_mask(m: torch.Tensor) -> np.ndarray:
        m_pil = Image.fromarray((m.squeeze(0).numpy() * 255).astype(np.uint8))
        return np.array(m_pil.resize((W, H), Image.NEAREST)) > 127

    pred = _resize_mask(pred_mask)
    gt   = _resize_mask(gt_mask)

    alpha = 0.5
    overlay = base.copy()
    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt
    overlay[tp] = overlay[tp] * (1 - alpha) + np.array([0,   200, 0  ], dtype=np.float32) * alpha
    overlay[fp] = overlay[fp] * (1 - alpha) + np.array([220, 0,   0  ], dtype=np.float32) * alpha
    overlay[fn] = overlay[fn] * (1 - alpha) + np.array([0,   80,  220], dtype=np.float32) * alpha
    return Image.fromarray(overlay.clip(0, 255).astype(np.uint8))


def boundary_overlap_image(
    base_img: Image.Image,
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    size: tuple,
    dilation: int = 3,
) -> Image.Image:
    """Visualise boundary alignment at display size.

    The base image is dimmed to 40 % so coloured boundary lines stand out:
      Green  — pred boundary pixel within `dilation` px of GT boundary (correct)
      Red    — pred boundary too far from GT boundary (false edge)
      Blue   — GT boundary too far from pred boundary (missed edge)
    """
    W, H = size

    def _resize_mask(m: torch.Tensor) -> torch.Tensor:
        m_np = (m.squeeze(0).numpy() * 255).astype(np.uint8)
        m_pil = Image.fromarray(m_np).resize((W, H), Image.NEAREST)
        arr = np.array(m_pil, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    p = _resize_mask(pred_mask)
    g = _resize_mask(gt_mask)
    p_boundary, g_boundary, dilated_p_boundary, dilated_g_boundary = _boundary_maps(p, g, dilation)

    p_b = p_boundary.squeeze().numpy().astype(bool)
    g_b = g_boundary.squeeze().numpy().astype(bool)
    d_p = dilated_p_boundary.squeeze().numpy().astype(bool)
    d_g = dilated_g_boundary.squeeze().numpy().astype(bool)

    base = np.array(base_img.resize(size, Image.LANCZOS), dtype=np.float32) * 0.4
    hit  = p_b & d_g          # pred boundary near GT boundary — correct edge
    miss = p_b & ~d_g         # pred boundary far from GT boundary — false edge
    skip = g_b & ~d_p         # GT boundary far from pred boundary — missed edge

    base[hit]  = base[hit]  / 0.4 * 0.3 + np.array([0,   210, 60 ], dtype=np.float32) * 0.7
    base[miss] = base[miss] / 0.4 * 0.3 + np.array([220, 30,  30 ], dtype=np.float32) * 0.7
    base[skip] = base[skip] / 0.4 * 0.3 + np.array([30,  100, 220], dtype=np.float32) * 0.7
    return Image.fromarray(base.clip(0, 255).astype(np.uint8))


# ──────────────────────────────────────────────────────────────────────────────
# Model loaders (lazy, singleton)
# ──────────────────────────────────────────────────────────────────────────────

_clip_model = None
_clip_proc = None

def get_clip(device):
    global _clip_model, _clip_proc
    if _clip_model is None:
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        _clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _clip_model, _clip_proc


_dino_model = None

def get_dino(device):
    global _dino_model
    if _dino_model is None:
        _dino_model = timm.create_model("vit_small_patch8_224.dino", pretrained=True).to(device).eval()
    return _dino_model


_segformer_model = None
_segformer_proc = None
_GARMENT_LABEL_IDS = None

def get_segformer(device):
    global _segformer_model, _segformer_proc, _GARMENT_LABEL_IDS
    if _segformer_model is None:
        _segformer_proc = SegformerImageProcessor.from_pretrained(
            "mattmdjaga/segformer_b2_clothes"
        )
        _segformer_model = SegformerForSemanticSegmentation.from_pretrained(
            "mattmdjaga/segformer_b2_clothes"
        ).to(device).eval()
        # Labels that correspond to upper-body garment classes in this model
        # (0=background, 4=upper-clothes, 5=dress, 7=coat, 8=pants, 11=skirt, etc.)
        _GARMENT_LABEL_IDS = {4, 5, 7}
    return _segformer_model, _segformer_proc, _GARMENT_LABEL_IDS


_lpips_metric = None

def get_lpips(device):
    global _lpips_metric
    if _lpips_metric is None:
        _lpips_metric = LPIPS(net_type="vgg").to(device)
    return _lpips_metric


# ──────────────────────────────────────────────────────────────────────────────
# Per-metric computations
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_lpips(gen_t: torch.Tensor, gt_t: torch.Tensor, device) -> float:
    """LPIPS between two [C,H,W] float32 [0,1] tensors."""
    g = gen_t.unsqueeze(0).to(device) * 2 - 1
    t = gt_t.unsqueeze(0).to(device) * 2 - 1
    return get_lpips(device)(g, t).item()


@torch.no_grad()
def compute_clip_sim(gen_img: Image.Image, gt_img: Image.Image, device) -> float:
    model, proc = get_clip(device)
    inputs = proc(images=[gen_img, gt_img], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    feats = model.get_image_features(**inputs)
    return cosine_sim(feats[0], feats[1])


@torch.no_grad()
def compute_dino_sim(
    gen_t: torch.Tensor, gt_t: torch.Tensor,
    mask_t: torch.Tensor, device
) -> float:
    """DINO cosine similarity on the background (non-garment) region."""
    model = get_dino(device)
    gen_bg = (apply_mask_outside(gen_t.to(device), mask_t.to(device)))
    gt_bg  = (apply_mask_outside(gt_t.to(device), mask_t.to(device)))
    size = (224, 224)
    gen_resized = F.interpolate(gen_bg.unsqueeze(0), size=size, mode="bilinear", align_corners=False)
    gt_resized  = F.interpolate(gt_bg.unsqueeze(0), size=size, mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    gen_n = (gen_resized - mean) / std
    gt_n  = (gt_resized  - mean) / std
    feat_gen = model.forward_features(gen_n)
    feat_gt  = model.forward_features(gt_n)
    if isinstance(feat_gen, dict):
        feat_gen = feat_gen["x_norm_clstoken"] if "x_norm_clstoken" in feat_gen else feat_gen["x"][:, 0]
        feat_gt  = feat_gt["x_norm_clstoken"]  if "x_norm_clstoken" in feat_gt  else feat_gt["x"][:, 0]
    elif feat_gen.dim() == 3:
        feat_gen = feat_gen[:, 0]
        feat_gt  = feat_gt[:, 0]
    return cosine_sim(feat_gen[0], feat_gt[0])


@torch.no_grad()
def segment_garment_mask(img_pil: Image.Image, device) -> torch.Tensor:
    """Run SegFormer and return a binary garment mask as [1, H, W] float32."""
    model, proc, label_ids = get_segformer(device)
    inputs = proc(images=img_pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits
    H, W = img_pil.size[1], img_pil.size[0]
    upsampled = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    pred = upsampled.argmax(dim=1).squeeze(0)
    garment = torch.zeros_like(pred, dtype=torch.float32)
    for lid in label_ids:
        garment = garment + (pred == lid).float()
    return garment.clamp(0, 1).unsqueeze(0)


# ──────────────────────────────────────────────────────────────────────────────
# Build file pairs from result_dir and data_dir
# ──────────────────────────────────────────────────────────────────────────────

def find_pairs(result_dir: str, data_dir: str):
    """
    Match generated images in result_dir to ground-truth target images.
    Convention from inference_fit.py: {stem}_result.png → target/{stem}.png
    Returns list of (gen_path, gt_path, mask_path_or_None, sample_id)
    """
    result_dir = Path(result_dir)
    data_dir = Path(data_dir)
    target_dir = data_dir / "target"
    mask_dir = data_dir / "agnostic-mask"

    pairs = []
    for gen_path in sorted(result_dir.glob("*_result.png")):
        stem = gen_path.stem.replace("_result", "")
        gt_candidates = list(target_dir.glob(f"{stem}.*"))
        if not gt_candidates:
            # Fall back: same stem in data_dir (VITON-HD layout)
            gt_candidates = list((data_dir / "test" / "image").glob(f"{stem}.*"))
        if not gt_candidates:
            continue
        gt_path = gt_candidates[0]
        # agnostic inpainting mask (torso region)
        mask_path = None
        for d in [mask_dir, data_dir / "test" / "agnostic-mask"]:
            cands = list(d.glob(f"{stem}.*")) if d.exists() else []
            if cands:
                mask_path = cands[0]
                break
        pairs.append((gen_path, gt_path, mask_path, stem))

    return pairs


def load_measurements(data_dir: str) -> dict:
    meas_path = Path(data_dir) / "measurements.json"
    if not meas_path.exists():
        return {}
    with open(meas_path) as f:
        records = json.load(f)
    return {Path(r["target"]).stem: r for r in records}


# ──────────────────────────────────────────────────────────────────────────────
# Side-by-side visualisation
# ──────────────────────────────────────────────────────────────────────────────

def save_visual(
    gen: Image.Image, gt: Image.Image, sample_id: str,
    cloth: Optional[Image.Image], person: Optional[Image.Image],
    metrics: dict, out_dir: Path,
    pred_mask: Optional[torch.Tensor] = None,
    gt_mask: Optional[torch.Tensor] = None,
):
    size = (256, 342)
    cols = []
    if person is not None:
        cols.append(("Input person", person.resize(size, Image.LANCZOS)))
    if cloth is not None:
        cols.append(("Garment", cloth.resize(size, Image.LANCZOS)))
    cols.append(("Generated", gen.resize(size, Image.LANCZOS)))
    cols.append(("Ground truth", gt.resize(size, Image.LANCZOS)))
    if pred_mask is not None and gt_mask is not None:
        cols.append(("Mask overlap", mask_overlap_image(gen, pred_mask, gt_mask, size)))
        cols.append(("Boundary", boundary_overlap_image(gen, pred_mask, gt_mask, size)))

    W, H = size
    header = 20
    canvas = Image.new("RGB", (W * len(cols), H + header), (240, 240, 240))
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(canvas)
    for i, (label, img) in enumerate(cols):
        canvas.paste(img, (i * W, header))
        draw.text((i * W + 4, 2), label, fill=(40, 40, 40))

    priority_keys = ("ssim", "psnr", "lpips", "clip_sim", "garment_iou", "garment_boundary_f1")
    ordered = [(k, metrics[k]) for k in priority_keys if k in metrics]
    ordered += [(k, v) for k, v in metrics.items() if k not in priority_keys]
    metric_str = "  ".join(
        f"{k}={v:.3f}" for k, v in ordered
        if isinstance(v, float) and not np.isnan(v)
    )
    footer_h = 18
    footer = Image.new("RGB", (W * len(cols), footer_h), (220, 220, 220))
    d2 = ImageDraw.Draw(footer)
    d2.text((4, 2), metric_str, fill=(40, 40, 40))
    full = Image.new("RGB", (W * len(cols), H + header + footer_h), (240, 240, 240))
    full.paste(canvas, (0, 0))
    full.paste(footer, (0, H + header))
    full.save(out_dir / f"{sample_id}.png")


# ──────────────────────────────────────────────────────────────────────────────
# FID / KID collection pass
# ──────────────────────────────────────────────────────────────────────────────

def compute_fid_kid(
    gen_tensors: list, gt_tensors: list, device
) -> dict:
    """Compute FID and KID from lists of [C,H,W] float32 [0,1] tensors."""
    results = {}
    if len(gen_tensors) < 2:
        return {"fid": float("nan"), "kid_mean": float("nan"), "kid_std": float("nan")}

    def to_uint8_resized(tensors, size=(299, 299)):
        """Stack float32 [0,1] tensors → uint8 [N,C,H,W] resized for Inception."""
        batch = torch.stack(tensors)  # [N, C, H, W] float32 [0,1]
        resized = F.interpolate(batch, size=size, mode="bilinear", align_corners=False)
        return (resized * 255).clamp(0, 255).to(torch.uint8)

    gen_b = to_uint8_resized(gen_tensors)
    gt_b  = to_uint8_resized(gt_tensors)

    try:
        fid_metric = FID(feature=2048).to(device)
        fid_metric.update(gen_b.to(device), real=False)
        fid_metric.update(gt_b.to(device),  real=True)
        results["fid"] = fid_metric.compute().item()
    except Exception as e:
        print(f"FID computation failed: {e}")
        results["fid"] = float("nan")

    try:
        kid_metric = KID(feature=2048, subset_size=min(len(gen_tensors), 50)).to(device)
        kid_metric.update(gen_b.to(device), real=False)
        kid_metric.update(gt_b.to(device),  real=True)
        kid_mean, kid_std = kid_metric.compute()
        results["kid_mean"] = kid_mean.item()
        results["kid_std"]  = kid_std.item()
    except Exception as e:
        print(f"KID computation failed: {e}")
        results["kid_mean"] = float("nan")
        results["kid_std"]  = float("nan")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Garment mask metric chart
# ──────────────────────────────────────────────────────────────────────────────

def save_mask_metrics_chart(per_sample: list, out_dir: Path):
    """Bar chart of per-sample garment_iou and garment_boundary_f1, saved as mask_metrics.png."""
    ids  = [r["sample_id"] for r in per_sample]
    ious = [r.get("garment_iou", float("nan")) for r in per_sample]
    f1s  = [r.get("garment_boundary_f1", float("nan")) for r in per_sample]

    valid = [not np.isnan(v) for v in ious]
    if not any(valid):
        return

    x = np.arange(len(ids))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(6, len(ids) * 0.6), 4))
    ax.bar(x - width / 2, ious, width, label="IoU",         color="#4C72B0")
    ax.bar(x + width / 2, f1s,  width, label="Boundary F1", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Garment mask quality per sample")
    ax.axhline(np.nanmean(ious), color="#4C72B0", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(np.nanmean(f1s),  color="#DD8452", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.legend()
    fig.tight_layout()
    chart_path = out_dir / "mask_metrics.png"
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)
    print(f"Mask metrics chart → {chart_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Counterfactual sensitivity (measurement controllability)
# ──────────────────────────────────────────────────────────────────────────────

MEASUREMENT_NAMES = [
    "body_bust", "body_height", "body_hips", "body_waist",
    "garment_bust", "garment_length", "garment_sleeve_length",
]


def _garment_extent(mask: torch.Tensor):
    """(area, bbox_height, bbox_width) of a [1,H,W] binary mask, as fractions of the image size."""
    m = mask.squeeze(0) > 0.5
    H, W = m.shape
    area = m.float().mean().item()
    if not m.any():
        return area, 0.0, 0.0
    rows = torch.where(m.any(dim=1))[0]
    cols = torch.where(m.any(dim=0))[0]
    return area, (rows[-1] - rows[0] + 1).item() / H, (cols[-1] - cols[0] + 1).item() / W


def counterfactual_from_images(cf_dir: str, device: torch.device) -> dict:
    """
    Analyse the +/- sigma images written by `inference_*.py --counterfactual`
    (<cf_dir>/<dimension>/<stem>_{plus,minus}.png). Everything except one measurement is identical
    between the pair (same inputs, seed and steps), so differences in the SegFormer garment mask are
    attributable to that measurement. Per dimension, over samples:
      mean/std_mask_area_delta : |area(+) - area(-)|             (magnitude of the response)
      signed_*                 : value(+) - value(-) for area, bbox height and bbox width
                                 (direction; positive = larger measurement -> larger garment)
    """
    cf_dir = Path(cf_dir)
    meta_path = cf_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    results = {"sigma": meta.get("sigma")}
    for name in MEASUREMENT_NAMES:
        dim_dir = cf_dir / name
        if not dim_dir.exists():
            continue
        abs_area, d_area, d_h, d_w = [], [], [], []
        for plus_path in sorted(dim_dir.glob("*_plus.png")):
            minus_path = plus_path.with_name(plus_path.name.replace("_plus.png", "_minus.png"))
            if not minus_path.exists():
                continue
            a_p, h_p, w_p = _garment_extent(segment_garment_mask(load_image_rgb(str(plus_path)), device))
            a_m, h_m, w_m = _garment_extent(segment_garment_mask(load_image_rgb(str(minus_path)), device))
            abs_area.append(abs(a_p - a_m))
            d_area.append(a_p - a_m)
            d_h.append(h_p - h_m)
            d_w.append(w_p - w_m)
        n = len(abs_area)
        results[name] = {
            "n_samples": n,
            "mean_mask_area_delta": float(np.mean(abs_area)) if n else float("nan"),
            "std_mask_area_delta":  float(np.std(abs_area))  if n else float("nan"),
            "signed_area_delta":    float(np.mean(d_area))   if n else float("nan"),
            "signed_height_delta":  float(np.mean(d_h))      if n else float("nan"),
            "signed_width_delta":   float(np.mean(d_w))      if n else float("nan"),
        }
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def print_counterfactual(cf: dict):
    print(f"\n── Counterfactual sensitivity (±{cf.get('sigma')}σ) ──────────────────────────────")
    print(f"  {'Dimension':<24} {'n':>3}  {'|Δarea|':>9} {'±std':>8}  {'Δarea':>9} {'Δheight':>9} {'Δwidth':>9}")
    for name, v in cf.items():
        if not isinstance(v, dict):
            continue
        print(f"  {name:<24} {v['n_samples']:>3}  {v['mean_mask_area_delta']:9.5f} {v['std_mask_area_delta']:8.5f}"
              f"  {v['signed_area_delta']:9.5f} {v['signed_height_delta']:9.5f} {v['signed_width_delta']:9.5f}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate IDM-VTON generated images.")
    p.add_argument("--result_dir", type=str, default=None,
                   help="Directory containing generated images (*_result.png). "
                        "Optional if only --cf_dir is given (counterfactual-only analysis).")
    p.add_argument("--data_dir",   type=str, default=None,
                   help="Dataset root (contains target/, agnostic-mask/, garment-mask/, measurements.json). "
                        "Optional if only --cf_dir is given.")
    p.add_argument("--output_dir", type=str, default="eval_output",
                   help="Where to write results.csv, summary.json, and visuals/.")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width",  type=int, default=768)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_visuals", action="store_true",
                   help="Skip saving side-by-side comparison images.")
    p.add_argument("--cf_dir", type=str, default=None,
                   help="Counterfactual image dir written by inference_*.py --counterfactual "
                        "(<output_dir>/cf). Enables the measurement controllability analysis.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "visuals"
    if not args.no_visuals:
        vis_dir.mkdir(exist_ok=True)

    if args.cf_dir and not (args.result_dir and args.data_dir):
        print("Analysing counterfactual images …")
        cf = counterfactual_from_images(args.cf_dir, device)
        cf_path = out_dir / "counterfactual.json"
        cf_path.write_text(json.dumps(cf, indent=2))
        print(f"Counterfactual results → {cf_path}")
        print_counterfactual(cf)
        return
    if not (args.result_dir and args.data_dir):
        sys.exit("--result_dir and --data_dir are required unless only --cf_dir is given.")

    pairs = find_pairs(args.result_dir, args.data_dir)
    if not pairs:
        print(f"No matched pairs found in {args.result_dir} / {args.data_dir}")
        sys.exit(1)
    print(f"Found {len(pairs)} image pairs.")

    measurements = load_measurements(args.data_dir)
    size = (args.width, args.height)  # PIL uses (W, H)

    per_sample = []
    gen_tensors_for_fid = []
    gt_tensors_for_fid  = []

    for gen_path, gt_path, mask_path, sample_id in tqdm(pairs, desc="Per-sample metrics"):
        gen_img = load_image_rgb(str(gen_path), size=size)
        gt_img  = load_image_rgb(str(gt_path),  size=size)

        gen_arr = np.array(gen_img)
        gt_arr  = np.array(gt_img)

        gen_t = pil_to_tensor_01(gen_img)
        gt_t  = pil_to_tensor_01(gt_img)

        # FID/KID collection
        gen_tensors_for_fid.append(gen_t.clone())
        gt_tensors_for_fid.append(gt_t.clone())

        # Inpainting mask (body region, for masked metrics)
        if mask_path is not None:
            body_mask_pil = Image.open(str(mask_path)).convert("L").resize(size, Image.NEAREST)
            body_mask = mask_to_binary(body_mask_pil)
        else:
            body_mask = torch.ones(1, args.height, args.width)

        row = {"sample_id": sample_id}

        # ── Pixel metrics ──────────────────────────────────────────────────────
        row["ssim"]  = ssim_numpy(gen_arr, gt_arr)
        row["psnr"]  = psnr_numpy(gen_arr, gt_arr)
        row["lpips"] = compute_lpips(gen_t, gt_t, device)

        # ── Garment region metrics (inside body mask) ──────────────────────────
        gen_garment = apply_mask_inside(gen_t, body_mask)
        gt_garment  = apply_mask_inside(gt_t,  body_mask)
        row["garment_ssim"]  = ssim_numpy(
            (gen_garment.permute(1, 2, 0).numpy() * 255).astype(np.uint8),
            (gt_garment.permute(1, 2, 0).numpy() * 255).astype(np.uint8),
        )
        row["garment_lpips"] = compute_lpips(gen_garment, gt_garment, device)

        # ── Garment mask IoU ───────────────────────────────────────────────────
        # Both masks come from the same SegFormer so the metric is symmetric:
        # IoU = 1 when gen == GT, regardless of what disk annotation files contain.
        # The disk mask (gm_path) is ignored for the primary metric.
        pred_gm_mask_for_vis = None
        gt_gm_mask_for_vis   = None
        pred_gm_mask = segment_garment_mask(gen_img, device)
        gt_gm_mask   = segment_garment_mask(gt_img,  device)
        if pred_gm_mask is not None and gt_gm_mask is not None:
            row["garment_iou"]         = iou_binary(pred_gm_mask, gt_gm_mask)
            row["garment_boundary_f1"] = boundary_f1(pred_gm_mask, gt_gm_mask)
            pred_gm_mask_for_vis = pred_gm_mask.cpu()
            gt_gm_mask_for_vis   = gt_gm_mask.cpu()
        else:
            row["garment_iou"]         = float("nan")
            row["garment_boundary_f1"] = float("nan")

        # ── CLIP similarity ────────────────────────────────────────────────────
        row["clip_sim"] = compute_clip_sim(gen_img, gt_img, device)

        # ── DINO similarity (background region) ───────────────────────────────
        row["dino_sim"] = compute_dino_sim(gen_t, gt_t, body_mask, device)

        # ── Save visual ────────────────────────────────────────────────────────
        if not args.no_visuals:
            cloth_path_cands = list(Path(args.data_dir).glob(f"cloth/{sample_id}.*"))
            person_path_cands = list(Path(args.data_dir).glob(f"person/{sample_id}.*"))
            cloth_img  = load_image_rgb(str(cloth_path_cands[0]),  size=size) if cloth_path_cands  else None
            person_img = load_image_rgb(str(person_path_cands[0]), size=size) if person_path_cands else None
            visual_metrics = {
                k: row[k] for k in ("ssim", "psnr", "lpips", "clip_sim", "garment_iou", "garment_boundary_f1")
                if k in row and isinstance(row[k], float)
            }
            save_visual(
                gen_img, gt_img, sample_id, cloth_img, person_img, visual_metrics, vis_dir,
                pred_mask=pred_gm_mask_for_vis, gt_mask=gt_gm_mask_for_vis,
            )

        per_sample.append(row)

    # ── Garment mask metrics chart ─────────────────────────────────────────────
    if not args.no_visuals:
        save_mask_metrics_chart(per_sample, out_dir)

    # ── FID / KID ──────────────────────────────────────────────────────────────
    print("Computing FID / KID …")
    dist_metrics = compute_fid_kid(gen_tensors_for_fid, gt_tensors_for_fid, device)

    # ── Aggregate per-sample metrics ──────────────────────────────────────────
    agg = {}
    numeric_keys = [k for k in per_sample[0] if k != "sample_id"]
    for k in numeric_keys:
        vals = [r[k] for r in per_sample if isinstance(r[k], float) and not np.isnan(r[k])]
        if vals:
            agg[k]              = float(np.mean(vals))
            agg[f"{k}_std"]     = float(np.std(vals))
            agg[f"{k}_median"]  = float(np.median(vals))
    agg.update(dist_metrics)

    # ── Counterfactual sensitivity ─────────────────────────────────────────────
    if args.cf_dir:
        print("Analysing counterfactual images …")
        agg["counterfactual"] = counterfactual_from_images(args.cf_dir, device)

    # ── Write outputs ──────────────────────────────────────────────────────────
    import csv
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_sample[0].keys())
        writer.writeheader()
        writer.writerows(per_sample)
    print(f"Per-sample results → {csv_path}")

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"Summary → {summary_path}")

    print("\n── Summary ──────────────────────────────────────────────────────────")
    for k, v in sorted(agg.items()):
        if not k.endswith("_std") and not k.endswith("_median") and k != "counterfactual":
            std = agg.get(f"{k}_std", float("nan"))
            print(f"  {k:<28} {v:8.4f}  ± {std:.4f}")

    if "counterfactual" in agg:
        print_counterfactual(agg["counterfactual"])

if __name__ == "__main__":
    main()
