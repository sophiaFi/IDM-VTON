"""
evaluate.py — Evaluation pipeline for IDM-VTON (with optional measurement conditioning).

Usage — compare a result folder against ground truth:

    python evaluate.py \\
        --result_dir result_fit \\
        --data_dir data_mini_test \\
        --output_dir eval_output

    # With measurement counterfactual sensitivity test (requires checkpoint):
    python evaluate.py \\
        --result_dir result_fit \\
        --data_dir data_mini_test \\
        --checkpoint_dir output/checkpoint-500 \\
        --counterfactual \\
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

  Identity preservation:
    face_cos   — ArcFace-style face embedding cosine similarity (optional,
                 requires insightface; skipped when not installed)

  Pose consistency:
    pose_dist  — mean Euclidean distance of OpenPose keypoints (optional,
                 requires dwpose or openpose; skipped when not installed)

  Measurement controllability (--counterfactual flag):
    For each measurement dimension, vary it by ±2σ while keeping everything
    else fixed and measure how the output garment mask changes. -> execute on GPU

All per-sample results are written to <output_dir>/results.csv.
A summary JSON is written to <output_dir>/summary.json.
Side-by-side comparison images are written to <output_dir>/visuals/.
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# peft==0.6.0 LoraConfig doesn't accept kwargs added in later versions
# (e.g. loftq_config).  Patch __init__ to drop unknown keyword arguments.
try:
    from peft import LoraConfig as _LoraConfig
    import inspect as _inspect
    _lora_known = set(_inspect.signature(_LoraConfig.__init__).parameters)
    _lora_orig_init = _LoraConfig.__init__

    def _lora_permissive_init(self, *args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k in _lora_known}
        _lora_orig_init(self, *args, **kwargs)

    _LoraConfig.__init__ = _lora_permissive_init
except Exception:
    pass
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Optional imports: warn but don't crash if not available
# ──────────────────────────────────────────────────────────────────────────────

try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
    from torchmetrics.image import PeakSignalNoiseRatio as PSNR
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
    from torchmetrics.image.fid import FrechetInceptionDistance as FID
    from torchmetrics.image.kid import KernelInceptionDistance as KID
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False
    warnings.warn("torchmetrics not available — SSIM, PSNR, LPIPS, FID, KID skipped.")

try:
    import transformers
    from transformers import CLIPProcessor, CLIPModel
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    warnings.warn("transformers not available — CLIP similarity skipped.")

try:
    import timm
    HAS_DINO = True
except ImportError:
    HAS_DINO = False
    warnings.warn("timm not available — DINO similarity skipped.")

try:
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    HAS_SEGFORMER = True
except ImportError:
    HAS_SEGFORMER = False
    warnings.warn("transformers SegFormer not available — garment segmentation skipped.")

try:
    import insightface
    HAS_INSIGHTFACE = True
except ImportError:
    HAS_INSIGHTFACE = False

try:
    from skimage.metrics import structural_similarity
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


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
    if HAS_SKIMAGE:
        return structural_similarity(img1, img2, channel_axis=2, data_range=255)
    return float("nan")


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


def boundary_f1(mask_pred: torch.Tensor, mask_gt: torch.Tensor, dilation: int = 3) -> float:
    """Approximate boundary F1 via morphological dilation."""
    import torch.nn.functional as F
    kernel = torch.ones(1, 1, dilation * 2 + 1, dilation * 2 + 1, device=mask_pred.device)
    # ensure exactly 4D: [1, 1, H, W]
    p = mask_pred.float().view(1, 1, mask_pred.shape[-2], mask_pred.shape[-1])
    g = mask_gt.float().view(1, 1, mask_gt.shape[-2], mask_gt.shape[-1])
    p_boundary = (F.conv2d(p, kernel, padding=dilation) > 0).float() * (1 - p)
    g_boundary = (F.conv2d(g, kernel, padding=dilation) > 0).float() * (1 - g)
    # count boundary pixels of pred that overlap dilated gt boundary and vice-versa
    p_hits = (p_boundary * (F.conv2d(g, kernel, padding=dilation) > 0).float()).sum().item()
    g_hits = (g_boundary * (F.conv2d(p, kernel, padding=dilation) > 0).float()).sum().item()
    precision = p_hits / (p_boundary.sum().item() + 1e-8)
    recall = g_hits / (g_boundary.sum().item() + 1e-8)
    return 2 * precision * recall / (precision + recall + 1e-8)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a.flatten().unsqueeze(0), dim=1)
    b = F.normalize(b.flatten().unsqueeze(0), dim=1)
    return (a * b).sum().item()


# ──────────────────────────────────────────────────────────────────────────────
# Model loaders (lazy, singleton)
# ──────────────────────────────────────────────────────────────────────────────

_clip_model = None
_clip_proc = None

def get_clip(device):
    global _clip_model, _clip_proc
    if _clip_model is None and HAS_CLIP:
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        _clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _clip_model, _clip_proc


_dino_model = None

def get_dino(device):
    global _dino_model
    if _dino_model is None and HAS_DINO:
        _dino_model = timm.create_model("vit_small_patch8_224.dino", pretrained=True).to(device).eval()
    return _dino_model


_segformer_model = None
_segformer_proc = None
_GARMENT_LABEL_IDS = None

def get_segformer(device):
    global _segformer_model, _segformer_proc, _GARMENT_LABEL_IDS
    if _segformer_model is None and HAS_SEGFORMER:
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
    if _lpips_metric is None and HAS_TORCHMETRICS:
        _lpips_metric = LPIPS(net_type="vgg").to(device)
    return _lpips_metric


# ──────────────────────────────────────────────────────────────────────────────
# Per-metric computations
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_lpips(gen_t: torch.Tensor, gt_t: torch.Tensor, device) -> float:
    """LPIPS between two [C,H,W] float32 [0,1] tensors."""
    metric = get_lpips(device)
    if metric is None:
        return float("nan")
    g = gen_t.unsqueeze(0).to(device) * 2 - 1
    t = gt_t.unsqueeze(0).to(device) * 2 - 1
    return metric(g, t).item()


@torch.no_grad()
def compute_clip_sim(gen_img: Image.Image, gt_img: Image.Image, device) -> float:
    model, proc = get_clip(device)
    if model is None:
        return float("nan")
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
    if model is None:
        return float("nan")
    bg_mask = (1.0 - mask_t).to(device)
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
    if model is None:
        return None
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
    garment_mask_dir = data_dir / "garment-mask"

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
        # GT garment mask
        gm_path = None
        cands = list(garment_mask_dir.glob(f"{stem}.*")) if garment_mask_dir.exists() else []
        if cands:
            gm_path = cands[0]
        pairs.append((gen_path, gt_path, mask_path, gm_path, stem))

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
    metrics: dict, out_dir: Path
):
    size = (256, 342)
    cols = []
    if person is not None:
        cols.append(("Input person", person.resize(size, Image.LANCZOS)))
    if cloth is not None:
        cols.append(("Garment", cloth.resize(size, Image.LANCZOS)))
    cols.append(("Generated", gen.resize(size, Image.LANCZOS)))
    cols.append(("Ground truth", gt.resize(size, Image.LANCZOS)))

    W, H = size
    header = 20
    canvas = Image.new("RGB", (W * len(cols), H + header), (240, 240, 240))
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(canvas)
    for i, (label, img) in enumerate(cols):
        canvas.paste(img, (i * W, header))
        draw.text((i * W + 4, 2), label, fill=(40, 40, 40))

    metric_str = "  ".join(
        f"{k}={v:.3f}" for k, v in metrics.items()
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
    if not HAS_TORCHMETRICS or len(gen_tensors) < 2:
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
# Counterfactual sensitivity (measurement controllability)
# ──────────────────────────────────────────────────────────────────────────────

MEASUREMENT_NAMES = [
    "body_bust", "body_height", "body_hips", "body_waist",
    "garment_bust", "garment_length", "garment_sleeve_length",
]


def counterfactual_sensitivity(
    checkpoint_dir: str,
    data_dir: str,
    device: torch.device,
    sigma_range: float = 2.0,
    num_samples: int = 4,
    seed: int = 42,
    height: int = 682,
    width: int = 512,
) -> dict:
    """
    For each measurement dimension, generate images at +sigma_range and -sigma_range
    (all other dimensions held at their dataset values) and compute the mean absolute
    change in the segmented garment mask area.  Returns a dict keyed by dimension name.
    """
    import random
    from diffusers import AutoencoderKL
    from peft import PeftModel
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
    from torch.utils.data import DataLoader

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dtype = torch.float16 if "cuda" in str(device) else torch.float32

    base_model = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

    tokenizer   = AutoTokenizer.from_pretrained(base_model, subfolder="tokenizer", use_fast=False)
    tokenizer_2 = AutoTokenizer.from_pretrained(base_model, subfolder="tokenizer_2", use_fast=False)
    text_encoder   = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder").to(device, dtype=dtype)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(base_model, subfolder="text_encoder_2").to(device, dtype=dtype)
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae").to(device, dtype=dtype)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained("ckpt/image_encoder").to(device, dtype=dtype)

    unet_encoder = UNet2DConditionModel_ref.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet").to(device, dtype=dtype)
    unet_encoder.config.addition_embed_type = None
    unet_encoder.config["addition_embed_type"] = None
    unet_encoder.requires_grad_(False)

    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet", low_cpu_mem_usage=False, device_map=None)
    unet.config.encoder_hid_dim = image_encoder.config.hidden_size
    unet.config.encoder_hid_dim_type = "ip_image_proj"

    ip_path = "ckpt/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin"
    ip_state = torch.load(ip_path, map_location="cpu")
    import torch.nn as nn
    adapter_modules = nn.ModuleList(unet.attn_processors.values())
    adapter_modules.load_state_dict(ip_state["ip_adapter"], strict=True)

    image_proj_model = Resampler(
        dim=image_encoder.config.hidden_size, depth=4, dim_head=64, heads=20,
        num_queries=16, embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim, ff_mult=4,
    ).to(device, dtype=torch.float32)
    image_proj_model.load_state_dict(ip_state["image_proj"], strict=True)
    unet.encoder_hid_proj = image_proj_model

    # expand conv_in 9→13 (same logic as inference_fit.py)
    old = unet.conv_in
    conv_new = nn.Conv2d(13, old.out_channels, old.kernel_size, padding=old.padding)
    nn.init.zeros_(conv_new.weight)
    conv_new.weight.data[:, :old.in_channels] = old.weight.data
    conv_new.bias.data = old.bias.data
    unet.conv_in = conv_new
    unet.config["in_channels"] = 13
    unet.config.in_channels = 13

    unet = PeftModel.from_pretrained(unet, os.path.join(checkpoint_dir, "lora"))
    unet = unet.merge_and_unload()
    conv_state = torch.load(os.path.join(checkpoint_dir, "conv_in.pt"), map_location="cpu")
    unet.conv_in.load_state_dict(conv_state)
    unet = unet.to(device, dtype=dtype)
    unet.requires_grad_(False)

    menc = MeasurementEncoder(num_measurements=7, hidden_dim=256,
                               output_dim=unet.config.cross_attention_dim).to(device)
    menc.load_state_dict(torch.load(os.path.join(checkpoint_dir, "measurement_encoder.pt"), map_location="cpu"))
    menc.eval()

    pipe = TryonPipeline.from_pretrained(
        base_model, unet=unet, unet_encoder=unet_encoder, vae=vae,
        tokenizer=tokenizer, tokenizer_2=tokenizer_2,
        text_encoder=text_encoder, text_encoder_2=text_encoder_2,
        image_encoder=image_encoder, feature_extractor=CLIPImageProcessor(),
        torch_dtype=dtype, add_watermarker=False, safety_checker=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    dataset = FITDatasetWithMeasurements(data_root=data_dir, phase="test", size=(height, width))
    indices = list(range(min(num_samples, len(dataset))))
    loader = DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=1, shuffle=False)

    def _pad(t: torch.Tensor, multiple: int = 8):
        """Pad [B,C,H,W] so H and W are multiples of `multiple`; return (padded, (ph, pw))."""
        _, _, h, w = t.shape
        ph = (multiple - h % multiple) % multiple
        pw = (multiple - w % multiple) % multiple
        if ph == 0 and pw == 0:
            return t, (0, 0)
        return F.pad(t, (0, pw, 0, ph), mode="reflect"), (ph, pw)

    pipe_h = height + (8 - height % 8) % 8
    pipe_w = width  + (8 - width  % 8) % 8
    pad_h  = pipe_h - height
    pad_w  = pipe_w - width

    def generate(batch, measurements_override):
        person_images  = batch["person_image"].to(device, dtype=dtype)
        garment_images = batch["garment_image"].to(device, dtype=dtype)
        garment_clips  = batch["garment_image_clip"]
        pose_imgs      = batch["pose"].to(device, dtype=dtype)
        masks          = batch["mask"].to(device, dtype=dtype)
        prompts        = list(batch["text_prompts"])
        prompts_cloth  = list(batch["text_prompts_cloth"])
        bsz = person_images.shape[0]
        neg = ["monochrome, lowres, bad anatomy, worst quality, low quality"] * bsz
        image_embeds = torch.cat([garment_clips[i] for i in range(bsz)], dim=0).to(device, dtype=dtype)
        with torch.no_grad():
            prompt_embeds, neg_embeds, pooled, neg_pooled = pipe.encode_prompt(
                prompts, num_images_per_prompt=1,
                do_classifier_free_guidance=True, negative_prompt=neg,
            )
            (cloth_embeds, _, _, _) = pipe.encode_prompt(
                prompts_cloth, num_images_per_prompt=1,
                do_classifier_free_guidance=False, negative_prompt=neg,
            )
            m = measurements_override.to(device, dtype=torch.float32)
            m_tok = menc(m).to(dtype=prompt_embeds.dtype)
            null_tok = torch.zeros_like(m_tok)
            prompt_embeds = torch.cat([prompt_embeds, m_tok], dim=1)
            neg_embeds    = torch.cat([neg_embeds,    null_tok], dim=1)

            person_input = (person_images + 1.0) / 2.0
            if pad_h > 0 or pad_w > 0:
                person_input, _  = _pad(person_input)
                pose_imgs_in, _  = _pad(pose_imgs)
                masks_in, _      = _pad(masks)
                garment_in, _    = _pad(garment_images)
            else:
                pose_imgs_in = pose_imgs
                masks_in     = masks
                garment_in   = garment_images

            images = pipe(
                prompt_embeds=prompt_embeds, negative_prompt_embeds=neg_embeds,
                pooled_prompt_embeds=pooled, negative_pooled_prompt_embeds=neg_pooled,
                num_inference_steps=2, generator=torch.Generator(device).manual_seed(seed),
                strength=1.0, pose_img=pose_imgs_in, text_embeds_cloth=cloth_embeds,
                cloth=garment_in, mask_image=masks_in, image=person_input,
                height=pipe_h, width=pipe_w, guidance_scale=2.0,
                ip_adapter_image=image_embeds,
            )[0]

            if pad_h > 0 or pad_w > 0:
                images = [img.crop((0, 0, width, height)) for img in images]

        return images[0]

    dim_results = {name: [] for name in MEASUREMENT_NAMES}
    total_steps = num_samples * len(MEASUREMENT_NAMES)

    with tqdm(total=total_steps, desc="Counterfactual") as pbar:
      for batch in loader:
        m_base = batch["measurements"].clone()
        for dim_idx, dim_name in enumerate(MEASUREMENT_NAMES):
            pbar.set_postfix(dim=dim_name)
            m_plus  = m_base.clone(); m_plus[:, dim_idx]  += sigma_range
            m_minus = m_base.clone(); m_minus[:, dim_idx] -= sigma_range

            img_plus  = generate(batch, m_plus)
            img_minus = generate(batch, m_minus)
            pbar.update(1)

            # Measure change in garment mask area using SegFormer
            mask_plus  = segment_garment_mask(img_plus,  device)
            mask_minus = segment_garment_mask(img_minus, device)
            if mask_plus is not None and mask_minus is not None:
                area_plus  = mask_plus.mean().item()
                area_minus = mask_minus.mean().item()
                dim_results[dim_name].append(abs(area_plus - area_minus))

    return {
        name: {
            "mean_mask_area_delta": float(np.mean(v)) if v else float("nan"),
            "std_mask_area_delta":  float(np.std(v))  if v else float("nan"),
        }
        for name, v in dim_results.items()
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate IDM-VTON generated images.")
    p.add_argument("--result_dir", type=str, required=True,
                   help="Directory containing generated images (*_result.png).")
    p.add_argument("--data_dir",   type=str, required=True,
                   help="Dataset root (contains target/, agnostic-mask/, garment-mask/, measurements.json).")
    p.add_argument("--output_dir", type=str, default="eval_output",
                   help="Where to write results.csv, summary.json, and visuals/.")
    p.add_argument("--height", type=int, default=682)
    p.add_argument("--width",  type=int, default=512)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_visuals", action="store_true",
                   help="Skip saving side-by-side comparison images.")
    p.add_argument("--counterfactual", action="store_true",
                   help="Run measurement controllability test (requires --checkpoint_dir).")
    p.add_argument("--checkpoint_dir", type=str, default=None,
                   help="Checkpoint directory for counterfactual sensitivity test.")
    p.add_argument("--cf_samples", type=int, default=4,
                   help="Number of samples for counterfactual test.")
    p.add_argument("--cf_sigma", type=float, default=2.0,
                   help="Sigma offset for counterfactual perturbation.")
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

    for gen_path, gt_path, mask_path, gm_path, sample_id in tqdm(pairs, desc="Per-sample metrics"):
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

        # ── Garment mask IoU (from GT mask file or SegFormer) ──────────────────
        if gm_path is not None:
            gt_gm_pil   = Image.open(str(gm_path)).convert("L").resize(size, Image.NEAREST)
            gt_gm_mask  = mask_to_binary(gt_gm_pil)
            pred_gm_mask = segment_garment_mask(gen_img, device)
            if pred_gm_mask is not None and gt_gm_mask is not None:
                row["garment_iou"]        = iou_binary(pred_gm_mask, gt_gm_mask)
                row["garment_boundary_f1"] = boundary_f1(pred_gm_mask, gt_gm_mask)
            else:
                row["garment_iou"]        = float("nan")
                row["garment_boundary_f1"] = float("nan")
        else:
            row["garment_iou"]        = float("nan")
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
                k: row[k] for k in ("ssim", "psnr", "lpips", "clip_sim")
                if k in row and isinstance(row[k], float)
            }
            save_visual(gen_img, gt_img, sample_id, cloth_img, person_img, visual_metrics, vis_dir)

        per_sample.append(row)

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
    if args.counterfactual:
        if args.checkpoint_dir is None:
            print("--counterfactual requires --checkpoint_dir; skipping.")
        else:
            print("Running counterfactual sensitivity test …")
            cf_results = counterfactual_sensitivity(
                checkpoint_dir=args.checkpoint_dir,
                data_dir=args.data_dir,
                device=device,
                sigma_range=args.cf_sigma,
                num_samples=args.cf_samples,
                seed=args.seed,
                height=args.height,
                width=args.width,
            )
            agg["counterfactual"] = cf_results

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
        print("\n── Counterfactual sensitivity ───────────────────────────────────────")
        print(f"  {'Dimension':<30}  {'mean Δmask':>12}  {'std':>8}")
        for name, vals in agg["counterfactual"].items():
            print(f"  {name:<30}  {vals['mean_mask_area_delta']:12.6f}  {vals['std_mask_area_delta']:8.6f}")


if __name__ == "__main__":
    main()
