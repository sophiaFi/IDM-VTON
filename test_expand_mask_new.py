"""
Visual test for the bust expansion in get_mask_location (utils_mask.py).

For each sample the script runs the full get_mask_location pipeline twice:
  - baseline : no bust measurements passed (original behaviour)
  - expanded : body_bust_cm + garment_bust_cm passed (new expansion)

A three-panel row is saved per sample:
  left   — baseline mask overlaid on person (red tint)
  centre — expanded mask overlaid on person (red tint)
  right  — pixels added by the expansion (diff)

Cyan lines show shoulder→hip on each side; the yellow line shows the
keypoint-derived bust line used for the expansion.

Usage:
    python test_expand_mask_new.py [--data_root data_mini_test] [--n 4] [--device cuda]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "gradio_demo"))

from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from preprocess_fit import (
    _build_densepose_predictor,
    run_densepose_torso_mask,
)
from utils_mask import get_mask_location, get_bust_measurement_lines


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

PANEL_W, PANEL_H = 384, 512
LABEL_H = 24
PAD = 8
BG = (30, 30, 30)
TEXT_COLOR = (220, 220, 220)


def _label(text: str) -> Image.Image:
    img = Image.new("RGB", (PANEL_W, LABEL_H), BG)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((4, 4), text, fill=TEXT_COLOR, font=font)
    return img


def _draw_line_with_dots(draw, p0, p1, color, r=4, width=2):
    draw.line([p0, p1], fill=color, width=width)
    for x, y in [p0, p1]:
        draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=color)


def _panel(
    mask: Image.Image,
    person: Image.Image | None,
    title: str,
    bust_line: tuple | None = None,
    shoulder_hip_points: tuple | None = None,
) -> Image.Image:
    base = person.convert("RGB").resize((PANEL_W, PANEL_H)) if person else Image.new("RGB", (PANEL_W, PANEL_H), (80, 80, 80))
    mask_rs = mask.convert("L").resize((PANEL_W, PANEL_H))
    overlay = Image.new("RGB", (PANEL_W, PANEL_H), (220, 60, 60))
    alpha = mask_rs.point(lambda v: int(v * 0.55))
    composite = Image.composite(overlay, base, alpha)
    draw = ImageDraw.Draw(composite)
    if shoulder_hip_points is not None:
        ls, lh, rs, rh = shoulder_hip_points
        _draw_line_with_dots(draw, ls, lh, color=(0, 210, 210))
        _draw_line_with_dots(draw, rs, rh, color=(0, 210, 210))
    if bust_line is not None:
        p0 = (bust_line[0][0], bust_line[0][1])
        p1 = (bust_line[1][0], bust_line[1][1])
        _draw_line_with_dots(draw, p0, p1, color=(255, 230, 0))
    label = _label(title)
    out = Image.new("RGB", (PANEL_W, PANEL_H + LABEL_H), BG)
    out.paste(label, (0, 0))
    out.paste(composite, (0, LABEL_H))
    return out


def _bust_lines_panel(
    person: Image.Image | None,
    lines: dict,
    title: str,
) -> Image.Image:
    """Show person image with body_bust_px (green) and current_mask_bust_px (orange) lines.

    lines — the dict returned by get_bust_measurement_lines.
    """
    base = person.convert("RGB").resize((PANEL_W, PANEL_H)) if person else Image.new("RGB", (PANEL_W, PANEL_H), (80, 80, 80))
    draw = ImageDraw.Draw(base)

    bust_y = lines["bust_y"]

    # DensePose torso bounds — two vertical tick marks in magenta
    if lines["torso_bounds"] is not None:
        tx0, tx1 = lines["torso_bounds"]
        tick = 10
        for tx in (tx0, tx1):
            draw.line([(tx, bust_y - tick), (tx, bust_y + tick)], fill=(220, 0, 220), width=2)

    # body bust width — green
    (bx0, _), (bx1, _) = lines["body_line"]
    _draw_line_with_dots(draw, (bx0, bust_y), (bx1, bust_y), color=(60, 220, 60))

    # mask torso bust width — orange
    if lines["mask_line"] is not None:
        (mx0, _), (mx1, _) = lines["mask_line"]
        _draw_line_with_dots(draw, (mx0, bust_y), (mx1, bust_y), color=(255, 140, 0))

    label = _label(title)
    out = Image.new("RGB", (PANEL_W, PANEL_H + LABEL_H), BG)
    out.paste(label, (0, 0))
    out.paste(base, (0, LABEL_H))
    return out


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data_mini_test")
    parser.add_argument("--n", type=int, default=4, help="Number of samples to visualise")
    parser.add_argument("--out", default="test_expand_mask_new_output.png")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--category", default="upper_body",
                        choices=["upper_body", "lower_body", "dresses"])
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    measurements_path = os.path.join(data_root, "measurements.json")

    with open(measurements_path) as f:
        records = json.load(f)

    records = records[: args.n]

    print("Loading models...")
    parsing_model = Parsing(args.gpu_id)
    openpose_model = OpenPose(args.gpu_id)
    openpose_model.preprocessor.body_estimation.model.to(args.device)
    densepose_predictor, _ = _build_densepose_predictor(args.device)
    print("Models loaded.")

    rows = []
    for record in records:
        person_path = os.path.join(data_root, record["person"])
        person_img = Image.open(person_path).convert("RGB") if os.path.exists(person_path) else None
        small = person_img.resize((384, 512)) if person_img else Image.new("RGB", (384, 512))

        keypoints = openpose_model(small)
        model_parse, _ = parsing_model(small)
        torso_mask = run_densepose_torso_mask(densepose_predictor, small)

        # baseline — no measurements, original behaviour
        baseline_mask, _ = get_mask_location(
            "hd", args.category, model_parse, keypoints,
        )

        # expanded — with bust measurements and DensePose torso mask
        expanded_mask, _ = get_mask_location(
            "hd", args.category, model_parse, keypoints,
            body_bust_cm=record["body_bust"],
            garment_bust_cm=record["garment_bust"],
            densepose_torso_mask=torso_mask,
        )

        lines = get_bust_measurement_lines(
            keypoints, baseline_mask,
            densepose_torso_mask=torso_mask, height=PANEL_H,
        )
        bust_line  = lines["body_line"]
        bust_ratio = record["garment_bust"] / max(record["body_bust"], 1.0)
        bust_px    = bust_line[1][0] - bust_line[0][0]
        sh_points  = (lines["left_shoulder"], lines["left_hip"],
                      lines["right_shoulder"], lines["right_hip"])
        info = f"#{record['id']}  bust_ratio={bust_ratio:.2f}  bust_px={bust_px:.1f}"

        p_base = _panel(baseline_mask, person_img, f"baseline  {info}",
                        bust_line=bust_line, shoulder_hip_points=sh_points)
        p_exp  = _panel(expanded_mask, person_img, f"expanded  {info}",
                        bust_line=bust_line, shoulder_hip_points=sh_points)

        base_np = np.array(baseline_mask.resize((PANEL_W, PANEL_H))) > 127
        exp_np  = np.array(expanded_mask.resize((PANEL_W, PANEL_H))) > 127
        added   = ((exp_np & ~base_np).astype(np.uint8) * 255)
        p_diff  = _panel(Image.fromarray(added), person_img, "added region",
                         bust_line=bust_line, shoulder_hip_points=sh_points)

        # debug panel: green = body bust width, orange = mask torso bust width, magenta ticks = DensePose bounds
        p_bust = _bust_lines_panel(
            person_img, lines,
            f"bust lines  bust_ratio={bust_ratio:.2f}  green=body  orange=mask  magenta=torso",
        )

        row = Image.new("RGB", (PANEL_W * 4 + PAD * 3, PANEL_H + LABEL_H), BG)
        row.paste(p_base, (0, 0))
        row.paste(p_exp,  (PANEL_W + PAD, 0))
        row.paste(p_diff, (PANEL_W * 2 + PAD * 2, 0))
        row.paste(p_bust, (PANEL_W * 3 + PAD * 3, 0))
        rows.append(row)

    if not rows:
        print("No samples rendered.")
        return

    row_h = rows[0].height
    canvas = Image.new("RGB", (rows[0].width, row_h * len(rows) + PAD * (len(rows) - 1)), BG)
    for i, row in enumerate(rows):
        canvas.paste(row, (0, i * (row_h + PAD)))

    out_path = os.path.abspath(args.out)
    canvas.save(out_path)
    print(f"Saved: {out_path}  ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    main()
