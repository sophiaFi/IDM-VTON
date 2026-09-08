"""
Randomly samples a given number of datapoints from the FIT dataset.
Calculates mean and std for all measurements

Run with:
    python sample_data.py \
        --data-path "../../dataset/data" \
        --split "train" \
        --num-samples 500 \
        --buffer-size 10_000 \
        --resize \
        --output-name "data/test" \
"""
import json
import argparse
import numpy as np
from itertools import islice
from datasets import load_dataset
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm
from pathlib import Path


parser = argparse.ArgumentParser( description="Randomly sample entries from a Hugging Face dataset." )
parser.add_argument( "--data-path", type=str, required=True, help="Local folder containing the FIT dataset parquet files" )
parser.add_argument( "--split", type=str, choices=["train", "eval"], required=True, help="Dataset split to sample from" )
parser.add_argument( "--num-samples", type=int, required=True, help="Number of samples to extract" )
parser.add_argument( "--buffer-size", type=int, required=True, help="Buffer size for shuffling" )
parser.add_argument( "--resize", action="store_true", help="Resize images to 512x682" )
parser.add_argument( "--output-name", type=str, required=True, help="Name of the output dataset folder" )
args = parser.parse_args()

data_path = Path(args.data_path)
parquet_files = list(data_path.glob("*.parquet")) + list(data_path.glob(f"{args.split}-*.parquet"))
if not data_path.exists() or not parquet_files:
    print(f"Dataset not found at {data_path}, downloading from HuggingFace...")
    snapshot_download(
        repo_id="Yuanhao-Harry-Wang/fitvto-100k",
        repo_type="dataset",
        local_dir=str(data_path),
    )
else:
    print(f"Dataset found at {data_path}, skipping download")


dataset = load_dataset("parquet",
                       data_files={args.split: str(data_path / f"{args.split}-*.parquet")},
                       streaming=True,)

stream = dataset[args.split].shuffle(seed=42, buffer_size=args.buffer_size)

def resize_image(img, resize):
    if resize:
        return img.resize((512, 682), Image.LANCZOS)
    return img

samples = []
for example in tqdm(islice(stream, args.num_samples), total=args.num_samples):
    example_resized = {
        "cloth": resize_image(example["cloth"], args.resize),
        "target": resize_image(example["target"], args.resize),
        "person": resize_image(example["person"], args.resize),
        "body_bust": example["body_bust"],
        "body_height": example["body_height"],
        "body_hips": example["body_hips"],
        "body_waist": example["body_waist"],
        "garment_bust": example["garment_bust"],
        "garment_length": example["garment_length"],
        "garment_sleeve_length": example["garment_sleeve_length"],
        }
    samples.append(example_resized)

output_dir = Path(args.output_name)
cloth_dir = output_dir / "cloth"
target_dir = output_dir / "target"
person_dir = output_dir / "person"
cloth_dir.mkdir(parents=True, exist_ok=True)
target_dir.mkdir(parents=True, exist_ok=True)
person_dir.mkdir(parents=True, exist_ok=True)

measurements = []
for i, sample in enumerate(samples):
    filename = f"{i:05d}.png"
    sample["cloth"].save(cloth_dir / filename)
    sample["target"].save(target_dir / filename)
    sample["person"].save(person_dir / filename)

    measurements.append({
        "id": i, "cloth": f"cloth/{filename}",
        "target": f"target/{filename}",
        "person": f"person/{filename}",
        "body_bust": sample["body_bust"],
        "body_height": sample["body_height"],
        "body_hips": sample["body_hips"],
        "body_waist": sample["body_waist"],
        "garment_bust": sample["garment_bust"],
        "garment_length": sample["garment_length"],
        "garment_sleeve_length": sample["garment_sleeve_length"],
        })

with open(output_dir / "measurements.json", "w", encoding="utf-8") as f:
    json.dump( measurements, f, indent=2, ensure_ascii=False )
    print(f"Saved at: {output_dir}")
