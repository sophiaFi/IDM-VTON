"""Counterfactual measurement sweeps for the inference scripts.
"""

import json
import os

import numpy as np
import torch
from PIL import Image

from src.measurement_encoder import _MEASURE_MEAN, _MEASURE_STD

MEASUREMENT_NAMES = [
    "body_bust", "body_height", "body_hips", "body_waist",
    "garment_bust", "garment_length", "garment_sleeve_length",
]


# Measurements that get_mask_location uses to extend the agnostic mask.
MASK_DIMS = {"body_bust", "body_height", "garment_bust", "garment_length"}


def perturb_measurements(measurements: torch.Tensor, dim_idx: int, sigma: float):
    """Shift base measurement `dim_idx` by `sigma` std devs.

    Takes [B, 9] z-scored measurements; returns (z-scored [B, 9], raw [B, 9] in cm).
    """
    mean = _MEASURE_MEAN.to(measurements.device, torch.float64)
    std = _MEASURE_STD.to(measurements.device, torch.float64)
    raw = measurements.to(torch.float64) * std + mean
    raw[:, dim_idx] += sigma * std[dim_idx]
    raw[:, 7] = raw[:, 4] - raw[:, 0]  # bust_ease = garment_bust - body_bust
    raw[:, 8] = raw[:, 1] - raw[:, 5]  # hem_drop  = body_height - garment_length
    return ((raw - mean) / std).to(measurements.dtype), raw


def _rebuild_masks(dataloader, batch, raw):
    """Re-run the inference-time mask extension with the perturbed raw measurements."""
    dataset = dataloader.dataset
    dataset = getattr(dataset, "dataset", dataset)  # unwrap Subset
    masks = torch.stack([
        dataset.extended_mask(
            int(sample_idx),
            body_bust_cm=float(r[0]), body_height_cm=float(r[1]),
            garment_bust_cm=float(r[4]), garment_length_cm=float(r[5]),
        )
        for sample_idx, r in zip(batch["sample_index"], raw)
    ]).to(batch["mask"].dtype)
    return masks, batch["person_image"] * (1.0 - masks)


def add_counterfactual_args(parser):
    parser.add_argument(
        "--counterfactual", action="store_true",
        help="After each sample also generate +/- cf_sigma variants of every measurement in --cf_dims "
             "(saved raw under <output_dir>/cf/). Costs 2 * len(cf_dims) extra generations per sample.",
    )
    parser.add_argument("--cf_sigma", type=float, default=2.0,
                        help="Perturbation size in standard deviations of the measurement.")
    parser.add_argument("--cf_dims", nargs="+", default=MEASUREMENT_NAMES, choices=MEASUREMENT_NAMES,
                        help="Measurements to perturb (default: all seven base measurements).")
    parser.add_argument("--cf_only", action="store_true",
                        help="Skip the normal generation; only write the counterfactual variants.")
    parser.add_argument("--cf_mask", choices=["fixed", "extended"], default="fixed",
                        help="fixed: reuse the mask built from the original measurements (isolates the "
                             "measurement-token effect). extended: rebuild the inference-time extended mask "
                             "from the perturbed measurements (body_bust/body_height/garment_bust/"
                             "garment_length only; other dims keep the original mask).")


def iter_batches(dataloader, args):
    """Yield each original batch, followed by its counterfactual variants when enabled."""
    if args.counterfactual:
        cf_root = os.path.join(args.output_dir, "cf")
        os.makedirs(cf_root, exist_ok=True)
        with open(os.path.join(cf_root, "meta.json"), "w") as f:
            json.dump({"sigma": args.cf_sigma, "dims": list(args.cf_dims), "mask": args.cf_mask}, f, indent=2)

    for batch in dataloader:
        if batch is None:
            continue
        if not (args.counterfactual and args.cf_only):
            yield batch
        if not args.counterfactual:
            continue
        for name in args.cf_dims:
            idx = MEASUREMENT_NAMES.index(name)
            for variant, sign in (("plus", 1.0), ("minus", -1.0)):
                cf_batch = dict(batch)
                cf_batch["measurements"], raw = perturb_measurements(
                    batch["measurements"], idx, sign * args.cf_sigma
                )
                if args.cf_mask == "extended" and name in MASK_DIMS:
                    cf_batch["mask"], cf_batch["masked_person"] = _rebuild_masks(dataloader, batch, raw)
                    cf_batch["mask_cf"] = True
                cf_batch["cf_tag"] = (name, variant)
                yield cf_batch


def save_if_counterfactual(batch, images, output_dir) -> bool:
    """Save raw outputs of a counterfactual batch; returns True if the batch was one."""
    tag = batch.get("cf_tag")
    if tag is None:
        return False
    name, variant = tag
    out_dir = os.path.join(output_dir, "cf", name)
    os.makedirs(out_dir, exist_ok=True)
    for i, (img, fname) in enumerate(zip(images, batch["person_filename"])):
        stem = os.path.splitext(os.path.basename(fname))[0]
        img.save(os.path.join(out_dir, f"{stem}_{variant}.png"))
        if "mask_cf" in batch:
            mask_u8 = (batch["mask"][i, 0].float().cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(mask_u8, mode="L").save(os.path.join(out_dir, f"{stem}_{variant}_mask.png"))
    return True
