"""
Visual test for the bust and length expansion in get_mask_location (utils_mask.py).

For each sample the script runs the full get_mask_location pipeline twice:
  - baseline : no measurements passed (original behaviour)
  - expanded : all measurements passed (bust + length expansion)

A four-panel row is saved per sample:
  1 — baseline mask overlaid on person (red tint)
  2 — expanded mask overlaid on person (red tint)
  3 — pixels added by the expansion (diff)
  4 — debug: bust lines (green/orange) + length lines (blues/yellow) on person

Bust debug lines (panel 4):
  green        — body bust width from keypoints
  orange       — current garment mask bust width (torso-clipped)
  magenta ticks — DensePose torso column bounds

Length debug lines (panel 4):
  white dashes — shoulder keypoint row (measurement anchor)
  light blue   — garment top row (connected-component top touching shoulder)
  dark blue    — current garment hem row
  yellow dashes — target garment hem row
  purple       — hip keypoint span (used to derive px/cm scale)

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
from utils_mask import get_mask_location, get_bust_measurement_lines, get_length_measurement_lines


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


LEGEND_ITEM_H = 18   # height of each legend row
LEGEND_PAD    = 10   # horizontal padding between items
SWATCH_W      = 32   # width of the coloured swatch
SWATCH_H      = 10   # height of the coloured swatch
LEGEND_BG     = (20, 20, 20)


def _legend_strip(canvas_w: int) -> Image.Image:
    """Return a legend image that spans the full canvas width.

    Two rows of entries: panels 1-3 on the first row, debug panel on the second.
    Each entry is a small swatch (line / dashed line / tint block) + label text.
    """
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    # --- entry definitions: (label, draw_fn) ----------------------------------
    # draw_fn receives (draw, x0, y_mid) and paints the swatch in [x0, x0+SWATCH_W]

    def _swatch_solid(color, dashed=False, vertical_ticks=False):
        def _draw(draw, x0, y_mid):
            if vertical_ticks:
                cx = x0 + SWATCH_W // 2
                draw.line([(cx, y_mid - 6), (cx, y_mid + 6)], fill=color, width=2)
                draw.line([(cx - 8, y_mid - 6), (cx - 8, y_mid + 6)], fill=color, width=2)
                draw.line([(cx + 8, y_mid - 6), (cx + 8, y_mid + 6)], fill=color, width=2)
            elif dashed:
                x = x0
                end = x0 + SWATCH_W
                while x < end:
                    draw.line([(x, y_mid), (min(x + 5, end), y_mid)], fill=color, width=2)
                    x += 9
            else:
                r = 3
                draw.line([(x0, y_mid), (x0 + SWATCH_W, y_mid)], fill=color, width=2)
                draw.ellipse([(x0 - r, y_mid - r), (x0 + r, y_mid + r)], fill=color)
                draw.ellipse([(x0 + SWATCH_W - r, y_mid - r), (x0 + SWATCH_W + r, y_mid + r)], fill=color)
        return _draw

    def _swatch_tint(color, alpha=0.45):
        def _draw(draw, x0, y_mid):
            y0 = y_mid - SWATCH_H // 2
            y1 = y_mid + SWATCH_H // 2
            r = int(color[0] * alpha)
            g = int(color[1] * alpha)
            b = int(color[2] * alpha)
            draw.rectangle([(x0, y0), (x0 + SWATCH_W, y1)], fill=(r, g, b))
        return _draw

    row1 = [
        ("mask region",         _swatch_tint((220, 60, 60))),
        ("shoulder→hip",        _swatch_solid((0, 210, 210))),
        ("bust keypoint line",  _swatch_solid((255, 230, 0))),
    ]
    row2 = [
        ("body bust width",       _swatch_solid((60, 220, 60))),
        ("mask bust width",       _swatch_solid((255, 140, 0))),
        ("torso col. bounds",     _swatch_solid((220, 0, 220), vertical_ticks=True)),
        ("shoulder row (anchor)", _swatch_solid((255, 255, 255), dashed=True)),
        ("garment top (CC)",      _swatch_solid((130, 200, 255))),
        ("current hem",           _swatch_solid((30, 80, 220))),
        ("target hem",            _swatch_solid((255, 220, 0), dashed=True)),
        ("shoulder→hip (px/cm ref)", _swatch_solid((180, 80, 220))),
    ]

    def _measure_row(entries):
        total = LEGEND_PAD
        for label, _ in entries:
            bbox = font.getbbox(label)
            text_w = bbox[2] - bbox[0]
            total += SWATCH_W + 6 + text_w + LEGEND_PAD
        return total

    strip_h = LABEL_H + LEGEND_ITEM_H * 2 + LEGEND_PAD * 3
    img = Image.new("RGB", (canvas_w, strip_h), LEGEND_BG)
    draw = ImageDraw.Draw(img)

    # section header
    draw.text((LEGEND_PAD, LEGEND_PAD), "Legend:", fill=TEXT_COLOR, font=font)

    for row_idx, entries in enumerate([row1, row2]):
        y_mid = LEGEND_PAD + LABEL_H + row_idx * LEGEND_ITEM_H + LEGEND_ITEM_H // 2
        x = LEGEND_PAD
        for label, draw_fn in entries:
            draw_fn(draw, x, y_mid)
            x += SWATCH_W + 6
            draw.text((x, y_mid - 6), label, fill=TEXT_COLOR, font=font)
            bbox = font.getbbox(label)
            x += (bbox[2] - bbox[0]) + LEGEND_PAD

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


def _draw_dashed_hline(draw, y, x0, x1, color, dash=6, gap=4, width=1):
    x = int(x0)
    end = int(x1)
    while x < end:
        draw.line([(x, y), (min(x + dash, end), y)], fill=color, width=width)
        x += dash + gap


def _debug_panel(
    person: Image.Image | None,
    bust_lines: dict,
    length_lines: dict,
    title: str,
) -> Image.Image:
    """Combined debug panel showing both bust and length measurement geometry.

    Bust lines:
      green        — body bust width from keypoints
      orange       — garment mask bust width (torso-clipped)
      magenta ticks — DensePose torso column bounds

    Length lines:
      white dashed  — shoulder keypoint row (measurement anchor)
      light blue    — garment top row (connected-component top)
      dark blue     — current garment hem row
      yellow dashed — target garment hem row
      purple        — shoulder-to-hip reference segments (px/cm scale)
    """
    base = person.convert("RGB").resize((PANEL_W, PANEL_H)) if person else Image.new("RGB", (PANEL_W, PANEL_H), (80, 80, 80))
    draw = ImageDraw.Draw(base)

    # Torso column bounds: derive x span for horizontal length lines
    torso_bounds = length_lines["torso_bounds"]
    if torso_bounds is not None:
        line_x0, line_x1 = torso_bounds
    else:
        line_x0, line_x1 = 0, PANEL_W - 1

    # ---- Bust lines ----
    bust_y = bust_lines["bust_y"]

    if bust_lines["torso_bounds"] is not None:
        tx0, tx1 = bust_lines["torso_bounds"]
        tick = 10
        for tx in (tx0, tx1):
            draw.line([(tx, bust_y - tick), (tx, bust_y + tick)], fill=(220, 0, 220), width=2)

    (bx0, _), (bx1, _) = bust_lines["body_line"]
    _draw_line_with_dots(draw, (bx0, bust_y), (bx1, bust_y), color=(60, 220, 60))

    if bust_lines["mask_line"] is not None:
        (mx0, _), (mx1, _) = bust_lines["mask_line"]
        _draw_line_with_dots(draw, (mx0, bust_y), (mx1, bust_y), color=(255, 140, 0))

    # ---- Length lines ----
    shoulder_y         = length_lines["shoulder_y"]
    garment_top_y      = length_lines["garment_top_y"]
    garment_hem_y      = length_lines["garment_hem_y"]
    target_hem_y       = length_lines["target_garment_hem_y"]

    # shoulder anchor — white dashed
    _draw_dashed_hline(draw, shoulder_y, line_x0, line_x1, color=(255, 255, 255), width=2)
    draw.ellipse([(line_x0 - 3, shoulder_y - 3), (line_x0 + 3, shoulder_y + 3)], fill=(255, 255, 255))

    # garment top (CC top) — light blue solid
    _draw_line_with_dots(draw, (line_x0, garment_top_y), (line_x1, garment_top_y),
                         color=(130, 200, 255), r=3)

    # current hem — dark blue solid
    _draw_line_with_dots(draw, (line_x0, garment_hem_y), (line_x1, garment_hem_y),
                         color=(30, 80, 220), r=3)

    # target hem — yellow dashed
    target_y_clamped = int(round(min(target_hem_y, PANEL_H - 1)))
    _draw_dashed_hline(draw, target_y_clamped, line_x0, line_x1, color=(255, 220, 0), width=2)
    draw.ellipse([(line_x1 - 3, target_y_clamped - 3), (line_x1 + 3, target_y_clamped + 3)],
                 fill=(255, 220, 0))

    # shoulder-to-hip reference segments (px/cm scale) — purple
    (ls, lh), (rs, rh) = length_lines["shoulder_hip_line"]
    _draw_line_with_dots(draw, ls, lh, color=(180, 80, 220), r=3)
    _draw_line_with_dots(draw, rs, rh, color=(180, 80, 220), r=3)

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

        # expanded — with all measurements and DensePose torso mask
        expanded_mask, _ = get_mask_location(
            "hd", args.category, model_parse, keypoints,
            body_bust_cm=record["body_bust"],
            garment_bust_cm=record["garment_bust"],
            garment_length_cm=record["garment_length"],
            body_height_cm=record["body_height"],
            densepose_torso_mask=torso_mask,
            debug_dir="Mask_expansion_test_results"
        )

        bust_lines   = get_bust_measurement_lines(
            keypoints, baseline_mask,
            densepose_torso_mask=torso_mask, height=PANEL_H,
        )
        length_lines = get_length_measurement_lines(
            keypoints, model_parse, args.category,
            garment_length_cm=record["garment_length"],
            body_height_cm=record["body_height"],
            width=PANEL_W, densepose_torso_mask=torso_mask, height=PANEL_H,
        )

        bust_line  = bust_lines["body_line"]
        bust_ratio = record["garment_bust"] / max(record["body_bust"], 1.0)
        length_ratio = record["garment_length"] / max(record["body_height"], 1.0)
        sh_points  = (bust_lines["left_shoulder"], bust_lines["left_hip"],
                      bust_lines["right_shoulder"], bust_lines["right_hip"])
        info = (f"#{record['id']}  bust_ratio={bust_ratio:.2f}  "
                f"length_ratio={length_ratio:.2f}  px_per_cm={length_lines['px_per_cm']:.2f}")

        p_base = _panel(baseline_mask, person_img, f"baseline  {info}",
                        bust_line=bust_line, shoulder_hip_points=sh_points)
        p_exp  = _panel(expanded_mask, person_img, f"expanded  {info}",
                        bust_line=bust_line, shoulder_hip_points=sh_points)

        base_np = np.array(baseline_mask.resize((PANEL_W, PANEL_H))) > 127
        exp_np  = np.array(expanded_mask.resize((PANEL_W, PANEL_H))) > 127
        added   = ((exp_np & ~base_np).astype(np.uint8) * 255)
        p_diff  = _panel(Image.fromarray(added), person_img, "added region",
                         bust_line=bust_line, shoulder_hip_points=sh_points)

        # debug panel: bust + length measurement geometry
        p_debug = _debug_panel(
            person_img, bust_lines, length_lines, f"length ratio: {length_ratio}"
        )

        row = Image.new("RGB", (PANEL_W * 4 + PAD * 3, PANEL_H + LABEL_H), BG)
        row.paste(p_base,  (0, 0))
        row.paste(p_exp,   (PANEL_W + PAD, 0))
        row.paste(p_diff,  (PANEL_W * 2 + PAD * 2, 0))
        row.paste(p_debug, (PANEL_W * 3 + PAD * 3, 0))
        rows.append(row)

    if not rows:
        print("No samples rendered.")
        return

    row_h      = rows[0].height
    canvas_w   = rows[0].width
    legend     = _legend_strip(canvas_w)
    total_h    = row_h * len(rows) + PAD * (len(rows) - 1) + PAD + legend.height
    canvas     = Image.new("RGB", (canvas_w, total_h), BG)
    for i, row in enumerate(rows):
        canvas.paste(row, (0, i * (row_h + PAD)))
    canvas.paste(legend, (0, row_h * len(rows) + PAD * len(rows)))

    out_path = os.path.abspath(args.out)
    canvas.save(out_path)
    print(f"Saved: {out_path}  ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    main()
