"""
Compute mean and std for all 7 measurements over the full FIT train split.
Uses Welford's online algorithm so the full dataset never loads into memory.

Run with:
    python data/compute_measurement_stats.py --data-path "../../fit_data/train-*.parquet"
"""
import argparse
import math
from datasets import load_dataset
from tqdm import tqdm

MEASUREMENT_KEYS = [
    "body_bust",
    "body_height",
    "body_hips",
    "body_waist",
    "garment_bust",
    "garment_length",
    "garment_sleeve_length",
]

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", type=str, required=True,
                    help="Glob path to train parquet files, e.g. E:/fit_data/train-*.parquet")
args = parser.parse_args()

dataset = load_dataset("parquet", data_files={"train": args.data_path}, streaming=True)
stream = dataset["train"]

count = {k: 0 for k in MEASUREMENT_KEYS}
mean  = {k: 0.0 for k in MEASUREMENT_KEYS}
M2    = {k: 0.0 for k in MEASUREMENT_KEYS}

for example in tqdm(stream, desc="Computing stats"):
    for k in MEASUREMENT_KEYS:
        v = example[k]
        if v is None:
            continue
        count[k] += 1
        delta = v - mean[k]
        mean[k] += delta / count[k]
        delta2 = v - mean[k]
        M2[k] += delta * delta2

print("\n--- Stats (training split) ---")
print(f"{'Key':<30} {'N':>8} {'mean':>10} {'std':>10}")
for k in MEASUREMENT_KEYS:
    n = count[k]
    std = math.sqrt(M2[k] / n) if n > 1 else 0.0
    print(f"{k:<30} {n:>8} {mean[k]:>10.3f} {std:>10.3f}")

print("\nConstants for pasting into measurement_encoder.py:")
print("MEAN = torch.tensor([")
for k in MEASUREMENT_KEYS:
    print(f"    {mean[k]:.3f},  # {k}")
print("])")
print()
print("STD = torch.tensor([")
for k in MEASUREMENT_KEYS:
    n = count[k]
    std = math.sqrt(M2[k] / n) if n > 1 else 0.0
    print(f"    {std:.3f},  # {k}")
print("])")
