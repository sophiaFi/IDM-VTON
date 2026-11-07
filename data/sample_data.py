"""
Uniformly samples a given number of datapoints from the FIT dataset.

Row indices are drawn at random over the whole split and only the row groups
containing them are read, so no shuffle buffer is needed. Each sample is
written to disk as soon as it is read. Rows whose images cannot be decoded are
logged and replaced by another randomly drawn row.

Run with:
    python sample_data.py \
        --data-path "../../dataset/data" \
        --split "train" \
        --num-samples 10000 \
        --resize \
        --output-name "data/full"
"""
import io
import json
import argparse
import random
from collections import defaultdict
from bisect import bisect_right
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm
from pathlib import Path


IMAGE_COLUMNS = ["cloth", "target", "person"]
MEASUREMENT_COLUMNS = [
    "body_bust", "body_height", "body_hips", "body_waist",
    "garment_bust", "garment_length", "garment_sleeve_length",
]

parser = argparse.ArgumentParser( description="Randomly sample entries from the FIT parquet dataset." )
parser.add_argument( "--data-path", type=str, required=True, help="Local folder containing the FIT dataset parquet files" )
parser.add_argument( "--split", type=str, choices=["train", "eval"], required=True, help="Dataset split to sample from" )
parser.add_argument( "--num-samples", type=int, required=True, help="Number of samples to extract" )
parser.add_argument( "--resize", action="store_true", help="Resize images to 512x682" )
parser.add_argument( "--output-name", type=str, required=True, help="Name of the output dataset folder" )
parser.add_argument( "--seed", type=int, default=42, help="Seed for sampling row indices" )
args = parser.parse_args()

data_path = Path(args.data_path)
if not data_path.exists() or not list(data_path.glob(f"{args.split}-*.parquet")):
    print(f"Dataset not found at {data_path}, downloading from HuggingFace...")
    snapshot_download(
        repo_id="Yuanhao-Harry-Wang/fitvto-100k",
        repo_type="dataset",
        local_dir=str(data_path),
    )

parquet_files = sorted(data_path.glob(f"{args.split}-*.parquet"))

# Index every row group as (file, row_group, first_global_row) using metadata only.
row_groups = []
starts = []
total_rows = 0
for path in parquet_files:
    meta = pq.ParquetFile(path).metadata
    for rg in range(meta.num_row_groups):
        row_groups.append((path, rg))
        starts.append(total_rows)
        total_rows += meta.row_group(rg).num_rows
print(f"Found {len(parquet_files)} parquet files, {len(row_groups)} row groups, {total_rows} rows")

if args.num_samples > total_rows:
    raise ValueError(f"Requested {args.num_samples} samples but split only has {total_rows} rows")

rng = random.Random(args.seed)
chosen = rng.sample(range(total_rows), args.num_samples)


def load_image(cell):
    img = Image.open(io.BytesIO(cell["bytes"]))
    img.load()
    if args.resize:
        img = img.resize((512, 682), Image.LANCZOS)
    return img


def describe_cell(cell):
    if cell is None:
        return "None"
    data = cell.get("bytes")
    if not data:
        return f"empty bytes (path={cell.get('path')!r})"
    return f"{len(data)} bytes, starts with {data[:8]!r}"


output_dir = Path(args.output_name)
for sub in IMAGE_COLUMNS:
    (output_dir / sub).mkdir(parents=True, exist_ok=True)

measurements = [None] * args.num_samples
pbar = tqdm(total=args.num_samples)
current_path, current_file = None, None


def process(assignments):
    """Read and save the given (global_row, out_id) pairs. Returns the ones that failed."""
    global current_path, current_file
    wanted = defaultdict(list)
    for global_row, out_id in assignments:
        rg_idx = bisect_right(starts, global_row) - 1
        wanted[rg_idx].append((global_row - starts[rg_idx], out_id, global_row))

    failed = []
    for rg_idx in sorted(wanted):
        path, rg = row_groups[rg_idx]
        if path != current_path:
            current_path, current_file = path, pq.ParquetFile(path)
        table = current_file.read_row_group(rg, columns=IMAGE_COLUMNS + MEASUREMENT_COLUMNS)
        for row_in_group, out_id, global_row in wanted[rg_idx]:
            row = table.slice(row_in_group, 1).to_pylist()[0]
            try:
                images = {col: load_image(row[col]) for col in IMAGE_COLUMNS}
            except Exception as e:
                details = ", ".join(f"{col}: {describe_cell(row[col])}" for col in IMAGE_COLUMNS)
                tqdm.write(f"Skipping row {row_in_group} of row group {rg} in {path.name} "
                           f"(global row {global_row}): {type(e).__name__}: {e}\n    {details}")
                failed.append((global_row, out_id))
                continue
            filename = f"{out_id:05d}.png"
            for col, img in images.items():
                img.save(output_dir / col / filename)
            measurements[out_id] = {
                "id": out_id,
                **{col: f"{col}/{filename}" for col in IMAGE_COLUMNS},
                **{col: row[col] for col in MEASUREMENT_COLUMNS},
            }
            pbar.update(1)
        del table
    return failed


used = set(chosen)
pending = list(zip(chosen, range(args.num_samples)))
num_bad = 0
while pending:
    failed = process(pending)
    num_bad += len(failed)
    pending = []
    for _, out_id in failed:
        if len(used) >= total_rows:
            raise RuntimeError("Ran out of rows to draw replacements from")
        replacement = rng.randrange(total_rows)
        while replacement in used:
            replacement = rng.randrange(total_rows)
        used.add(replacement)
        pending.append((replacement, out_id))
pbar.close()
print(f"Replaced {num_bad} undecodable rows")

with open(output_dir / "measurements.json", "w", encoding="utf-8") as f:
    json.dump( measurements, f, indent=2, ensure_ascii=False )
    print(f"Saved at: {output_dir}")
