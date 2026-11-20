"""Counterfactual measurement sweeps for the inference scripts.

For every sample, each selected measurement is shifted by +/- cf_sigma standard deviations while
all other inputs, the seed and the number of steps stay fixed. The derived ease features
(bust_ease, hem_drop) are recomputed so the perturbed input stays consistent. Raw model outputs are
written to <output_dir>/cf/<dimension>/<stem>_{plus,minus}.png and analysed by evaluate.py --cf_dir.
"""

import json
import os

import numpy as np
import torch

from src.measurement_encoder import _MEASURE_MEAN, _MEASURE_STD

MEASUREMENT_NAMES = [
    "body_bust", "body_height", "body_hips", "body_waist",
    "garment_bust", "garment_length", "garment_sleeve_length",
]


def perturb_measurements(measurements: torch.Tensor, dim_idx: int, sigma: float) -> torch.Tensor:
    """Shift base measurement `dim_idx` by `sigma` std devs ([B, 9] z-scored in and out)."""
    mean = _MEASURE_MEAN.to(measurements.device, torch.float64)
    std = _MEASURE_STD.to(measurements.device, torch.float64)
    raw = measurements.to(torch.float64) * std + mean
    raw[:, dim_idx] += sigma * std[dim_idx]
    raw[:, 7] = raw[:, 4] - raw[:, 0]  # bust_ease = garment_bust - body_bust
    raw[:, 8] = raw[:, 1] - raw[:, 5]  # hem_drop  = body_height - garment_length
    return ((raw - mean) / std).to(measurements.dtype)


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


def iter_batches(dataloader, args):
    """Yield each original batch, followed by its counterfactual variants when enabled."""
    if args.counterfactual:
        cf_root = os.path.join(args.output_dir, "cf")
        os.makedirs(cf_root, exist_ok=True)
        with open(os.path.join(cf_root, "meta.json"), "w") as f:
            json.dump({"sigma": args.cf_sigma, "dims": list(args.cf_dims)}, f, indent=2)

    for batch in dataloader:
        if not (args.counterfactual and args.cf_only):
            yield batch
        if not args.counterfactual:
            continue
        for name in args.cf_dims:
            idx = MEASUREMENT_NAMES.index(name)
            for variant, sign in (("plus", 1.0), ("minus", -1.0)):
                cf_batch = dict(batch)
                cf_batch["measurements"] = perturb_measurements(
                    batch["measurements"], idx, sign * args.cf_sigma
                )
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
    for img, fname in zip(images, batch["person_filename"]):
        stem = os.path.splitext(os.path.basename(fname))[0]
        img.save(os.path.join(out_dir, f"{stem}_{variant}.png"))
    return True
