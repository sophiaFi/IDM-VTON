"""
Visual test for the bust and length expansion in get_mask_location (utils_mask.py).

For each sample the script runs get_mask_location twice:
  - baseline : no measurements passed (original behaviour)
  - expanded : all measurements passed (bust + length expansion)

A five-panel row is saved per sample:
  1 — baseline mask overlaid on person (red tint), no annotation lines
  2 — expanded mask overlaid on person (red tint), no annotation lines
  3 — pixels added by the expansion (diff), no annotation lines
  4 — bust debug: shoulder/hip connections, bust keypoints, torso borders, mask bust width
  5 — length debug: torso borders, garment lines, shoulder/hip rows, centre measure line

A row info banner above each row shows id, bust_ratio, length_ratio, px_per_cm.

Usage:
    python test_expand_mask_new.py [--data_root data_mini_test] [--n 4] [--device cuda]
"""

import argparse
import json
import math
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
# Layout constants
# ---------------------------------------------------------------------------

PANEL_W    = 384
PANEL_H    = 512
LABEL_H    = 24
ROW_INFO_H = 28
PAD        = 8
BG         = (30, 30, 30)
TEXT_COLOR = (220, 220, 220)


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

C_MASK_TINT    = (220, 60,  60)    # red tint — mask region overlay
C_SHOULDER_HIP = (0,  210, 210)    # cyan     — shoulder-hip connections + points
C_BUST_LINE    = (255, 230,  0)    # yellow   — bust keypoints connection
C_MASK_BUST    = (255, 140,  0)    # orange   — mask bust width
C_TORSO_BORDER = (220,   0, 220)   # magenta  — torso column vertical borders
C_GARMENT_TOP  = (130, 200, 255)   # lt blue  — garment top row
C_GARMENT_HEM  = (30,   80, 220)   # dk blue  — current garment hem
C_TARGET_HEM   = (255, 220,   0)   # yellow   — target garment hem (dashed)
C_SHOULDER_ROW = (255, 255, 255)   # white    — shoulder row (dashed)
C_HIP_ROW      = (100, 220, 100)   # lime     — hip row (dashed)
C_SH_CENTRE    = (180,  80, 220)   # purple   — shoulder-to-hip centre measure line


# ---------------------------------------------------------------------------
# Font helper
# ---------------------------------------------------------------------------

def _font(size: int = 12) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_line_with_dots(draw, p0, p1, color, r: int = 4, width: int = 2):
    draw.line([p0, p1], fill=color, width=width)
    for x, y in [p0, p1]:
        draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=color)


def _draw_dashed_hline(draw, y, x0, x1, color, dash: int = 6, gap: int = 4, width: int = 1):
    x = int(x0)
    end = int(x1)
    while x < end:
        draw.line([(x, y), (min(x + dash, end), y)], fill=color, width=width)
        x += dash + gap


def _draw_dashed_line(draw, p0, p1, color, dash: int = 6, gap: int = 4, width: int = 2):
    """Dashed line between two arbitrary points."""
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    length = max(1.0, math.hypot(dx, dy))
    ux, uy = dx / length, dy / length
    dist = 0.0
    drawing = True
    while dist < length:
        if drawing:
            ax = p0[0] + ux * dist
            ay = p0[1] + uy * dist
            bx = p0[0] + ux * min(dist + dash, length)
            by_ = p0[1] + uy * min(dist + dash, length)
            draw.line([(ax, ay), (bx, by_)], fill=color, width=width)
        dist += dash if drawing else gap
        drawing = not drawing


def _draw_vline(draw, x, y0, y1, color, width: int = 2):
    draw.line([(int(x), int(y0)), (int(x), int(y1))], fill=color, width=width)


# ---------------------------------------------------------------------------
# Label / banner builders
# ---------------------------------------------------------------------------

def _label(text: str) -> Image.Image:
    img = Image.new("RGB", (PANEL_W, LABEL_H), BG)
    draw = ImageDraw.Draw(img)
    draw.text((4, 4), text, fill=TEXT_COLOR, font=_font(12))
    return img


def _row_info_banner(text: str, width: int) -> Image.Image:
    img = Image.new("RGB", (width, ROW_INFO_H), (50, 50, 50))
    draw = ImageDraw.Draw(img)
    draw.text((8, 6), text, fill=(255, 255, 200), font=_font(14))
    return img


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------

def _mask_panel(mask: Image.Image, person: Image.Image | None, title: str) -> Image.Image:
    """Panels 1–3: red-tint mask overlay on person, no annotation lines."""
    base = (person.convert("RGB").resize((PANEL_W, PANEL_H))
            if person else Image.new("RGB", (PANEL_W, PANEL_H), (80, 80, 80)))
    mask_rs = mask.convert("L").resize((PANEL_W, PANEL_H))
    overlay = Image.new("RGB", (PANEL_W, PANEL_H), C_MASK_TINT)
    alpha   = mask_rs.point(lambda v: int(v * 0.55))
    composite = Image.composite(overlay, base, alpha)
    label = _label(title)
    out   = Image.new("RGB", (PANEL_W, PANEL_H + LABEL_H), BG)
    out.paste(label, (0, 0))
    out.paste(composite, (0, LABEL_H))
    return out


def _bust_debug_panel(
    person: Image.Image | None,
    bust_lines: dict,
    title: str,
) -> Image.Image:
    """Panel 4: bust-extension geometry.

    Draws:
      - Torso column borders as full-height vertical magenta lines
      - Left and right shoulder-hip connections (cyan, with endpoint dots)
      - Body bust width / bust keypoints connection (yellow, with endpoint dots)
      - Mask bust width (orange, with endpoint dots)
    """
    base = (person.convert("RGB").resize((PANEL_W, PANEL_H))
            if person else Image.new("RGB", (PANEL_W, PANEL_H), (80, 80, 80)))
    draw = ImageDraw.Draw(base)

    def _pt(key):
        return tuple(int(round(v)) for v in bust_lines[key])

    ls = _pt("left_shoulder")
    rs = _pt("right_shoulder")
    lh = _pt("left_hip")
    rh = _pt("right_hip")

    # Torso column borders — full height vertical lines
    torso_bounds = bust_lines["torso_bounds"]
    if torso_bounds is not None:
        tx0, tx1 = torso_bounds
        _draw_vline(draw, tx0, 0, PANEL_H - 1, C_TORSO_BORDER)
        _draw_vline(draw, tx1, 0, PANEL_H - 1, C_TORSO_BORDER)

    # Shoulder-hip connections (cyan)
    _draw_line_with_dots(draw, ls, lh, color=C_SHOULDER_HIP)
    _draw_line_with_dots(draw, rs, rh, color=C_SHOULDER_HIP)

    # Bust keypoints connection — body_line from bust_lines (yellow)
    bust_y = bust_lines["bust_y"]
    (bx0, _), (bx1, _) = bust_lines["body_line"]
    _draw_line_with_dots(draw, (int(round(bx0)), bust_y), (int(round(bx1)), bust_y), color=C_BUST_LINE)

    # Mask bust width (orange)
    if bust_lines["mask_line"] is not None:
        (mx0, _), (mx1, _) = bust_lines["mask_line"]
        _draw_line_with_dots(draw, (int(round(mx0)), bust_y), (int(round(mx1)), bust_y), color=C_MASK_BUST)

    label = _label(title)
    out   = Image.new("RGB", (PANEL_W, PANEL_H + LABEL_H), BG)
    out.paste(label, (0, 0))
    out.paste(base, (0, LABEL_H))
    return out


def _length_debug_panel(
    person: Image.Image | None,
    length_lines: dict,
    title: str,
) -> Image.Image:
    """Panel 5: length-extension geometry.

    Draws:
      - Torso column borders as full-height vertical magenta lines
      - Garment top row (light blue solid)
      - Current garment hem (dark blue solid)
      - Target garment hem (yellow dashed)
      - Dashed line between shoulder points (white)
      - Dashed line between hip points (lime green)
      - Shoulder-to-hip centre measure line (purple, shoulder centre → hip centre)
    """
    base = (person.convert("RGB").resize((PANEL_W, PANEL_H))
            if person else Image.new("RGB", (PANEL_W, PANEL_H), (80, 80, 80)))
    draw = ImageDraw.Draw(base)

    torso_bounds = length_lines["torso_bounds"]
    if torso_bounds is not None:
        line_x0, line_x1 = torso_bounds
    else:
        line_x0, line_x1 = 0, PANEL_W - 1

    # Torso column borders — full height vertical lines
    if torso_bounds is not None:
        _draw_vline(draw, line_x0, 0, PANEL_H - 1, C_TORSO_BORDER)
        _draw_vline(draw, line_x1, 0, PANEL_H - 1, C_TORSO_BORDER)

    # Garment top (light blue)
    garment_top_y = int(round(length_lines["garment_top_y"]))
    _draw_line_with_dots(draw, (line_x0, garment_top_y), (line_x1, garment_top_y), color=C_GARMENT_TOP, r=3)

    # Current garment hem (dark blue)
    garment_hem_y = int(round(length_lines["garment_hem_y"]))
    _draw_line_with_dots(draw, (line_x0, garment_hem_y), (line_x1, garment_hem_y), color=C_GARMENT_HEM, r=3)

    # Target garment hem (yellow dashed)
    target_hem_y = int(round(min(length_lines["target_garment_hem_y"], PANEL_H - 1)))
    _draw_dashed_hline(draw, target_hem_y, line_x0, line_x1, color=C_TARGET_HEM, width=2)
    r = 3
    draw.ellipse([(line_x1 - r, target_hem_y - r), (line_x1 + r, target_hem_y + r)], fill=C_TARGET_HEM)

    # Shoulder row: dashed line between the two shoulder points (white)
    ls = tuple(int(round(v)) for v in length_lines["left_shoulder"])
    rs = tuple(int(round(v)) for v in length_lines["right_shoulder"])
    _draw_dashed_line(draw, ls, rs, color=C_SHOULDER_ROW, width=2)
    for pt in [ls, rs]:
        draw.ellipse([(pt[0] - r, pt[1] - r), (pt[0] + r, pt[1] + r)], fill=C_SHOULDER_ROW)

    # Hip row: dashed line between the two hip points (lime green)
    lh = tuple(int(round(v)) for v in length_lines["left_hip"])
    rh = tuple(int(round(v)) for v in length_lines["right_hip"])
    _draw_dashed_line(draw, lh, rh, color=C_HIP_ROW, width=2)
    for pt in [lh, rh]:
        draw.ellipse([(pt[0] - r, pt[1] - r), (pt[0] + r, pt[1] + r)], fill=C_HIP_ROW)

    # Shoulder-to-hip centre measure line (purple)
    # — connects midpoint of shoulder pair to midpoint of hip pair
    sc = (int(round((ls[0] + rs[0]) / 2.0)), int(round((ls[1] + rs[1]) / 2.0)))
    hc = (int(round((lh[0] + rh[0]) / 2.0)), int(round((lh[1] + rh[1]) / 2.0)))
    _draw_line_with_dots(draw, sc, hc, color=C_SH_CENTRE, r=4)

    label = _label(title)
    out   = Image.new("RGB", (PANEL_W, PANEL_H + LABEL_H), BG)
    out.paste(label, (0, 0))
    out.paste(base, (0, LABEL_H))
    return out


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

LEGEND_ITEM_H = 18
LEGEND_PAD    = 10
SWATCH_W      = 32
SWATCH_H      = 10
LEGEND_BG     = (20, 20, 20)


def _legend_strip(canvas_w: int) -> Image.Image:
    f = _font(12)

    def _solid(color):
        def _draw(draw, x0, y_mid):
            draw.line([(x0, y_mid), (x0 + SWATCH_W, y_mid)], fill=color, width=2)
            r = 3
            draw.ellipse([(x0 - r, y_mid - r), (x0 + r, y_mid + r)], fill=color)
            draw.ellipse([(x0 + SWATCH_W - r, y_mid - r), (x0 + SWATCH_W + r, y_mid + r)], fill=color)
        return _draw

    def _dashed(color):
        def _draw(draw, x0, y_mid):
            x = x0
            end = x0 + SWATCH_W
            while x < end:
                draw.line([(x, y_mid), (min(x + 5, end), y_mid)], fill=color, width=2)
                x += 9
        return _draw

    def _vline(color):
        def _draw(draw, x0, y_mid):
            cx = x0 + SWATCH_W // 2
            draw.line([(cx, y_mid - 7), (cx, y_mid + 7)], fill=color, width=2)
        return _draw

    def _tint(color, alpha=0.45):
        def _draw(draw, x0, y_mid):
            y0 = y_mid - SWATCH_H // 2
            y1 = y_mid + SWATCH_H // 2
            draw.rectangle(
                [(x0, y0), (x0 + SWATCH_W, y1)],
                fill=(int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha)),
            )
        return _draw

    # Three rows: panels 1-3 | panel 4 bust | panel 5 length
    rows = [
        [("mask region",           _tint(C_MASK_TINT))],
        [
            ("shoulder-hip",       _solid(C_SHOULDER_HIP)),
            ("bust keypoints",     _solid(C_BUST_LINE)),
            ("mask bust width",    _solid(C_MASK_BUST)),
            ("torso borders",      _vline(C_TORSO_BORDER)),
        ],
        [
            ("garment top",        _solid(C_GARMENT_TOP)),
            ("current hem",        _solid(C_GARMENT_HEM)),
            ("target hem",         _dashed(C_TARGET_HEM)),
            ("shoulder row",       _dashed(C_SHOULDER_ROW)),
            ("hip row",            _dashed(C_HIP_ROW)),
            ("shoulder-hip centre", _solid(C_SH_CENTRE)),
        ],
    ]

    n_rows  = len(rows)
    strip_h = LABEL_H + LEGEND_ITEM_H * n_rows + LEGEND_PAD * (n_rows + 1)
    img     = Image.new("RGB", (canvas_w, strip_h), LEGEND_BG)
    draw    = ImageDraw.Draw(img)
    draw.text((LEGEND_PAD, LEGEND_PAD), "Legend:", fill=TEXT_COLOR, font=f)

    for row_idx, entries in enumerate(rows):
        y_mid = LEGEND_PAD + LABEL_H + row_idx * LEGEND_ITEM_H + LEGEND_ITEM_H // 2
        x = LEGEND_PAD
        for label_text, draw_fn in entries:
            draw_fn(draw, x, y_mid)
            x += SWATCH_W + 6
            draw.text((x, y_mid - 6), label_text, fill=TEXT_COLOR, font=f)
            bbox = f.getbbox(label_text)
            x += (bbox[2] - bbox[0]) + LEGEND_PAD

    return img


# ---------------------------------------------------------------------------
# Main
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

    data_root         = os.path.abspath(args.data_root)
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
        person_img  = Image.open(person_path).convert("RGB") if os.path.exists(person_path) else None
        small       = person_img.resize((384, 512)) if person_img else Image.new("RGB", (384, 512))

        keypoints  = openpose_model(small)
        model_parse, _ = parsing_model(small)
        torso_mask = run_densepose_torso_mask(densepose_predictor, small)

        baseline_mask, _ = get_mask_location("hd", args.category, model_parse, keypoints)
        expanded_mask, _ = get_mask_location(
            "hd", args.category, model_parse, keypoints,
            body_bust_cm=record["body_bust"],
            garment_bust_cm=record["garment_bust"],
            garment_length_cm=record["garment_length"],
            body_height_cm=record["body_height"],
            densepose_torso_mask=torso_mask,
            debug_dir="Mask_expansion_test_results",
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

        bust_ratio   = record["garment_bust"]   / max(record["body_bust"],   1.0)
        length_ratio = record["garment_length"] / max(record["body_height"], 1.0)

        # Diff: pixels added by the expansion
        base_np = np.array(baseline_mask.resize((PANEL_W, PANEL_H))) > 127
        exp_np  = np.array(expanded_mask.resize((PANEL_W, PANEL_H))) > 127
        added   = ((exp_np & ~base_np).astype(np.uint8) * 255)

        row_info = (
            f"#{record['id']}    "
            f"bust_ratio={bust_ratio:.2f}    "
            f"length_ratio={length_ratio:.2f}    "
            f"px_per_cm={length_lines['px_per_cm']:.2f}"
        )

        p1 = _mask_panel(baseline_mask,           person_img, "baseline")
        p2 = _mask_panel(expanded_mask,            person_img, "expanded")
        p3 = _mask_panel(Image.fromarray(added),   person_img, "added region")
        p4 = _bust_debug_panel(person_img,   bust_lines,   "bust extension")
        p5 = _length_debug_panel(person_img, length_lines, "length extension")

        canvas_w = PANEL_W * 5 + PAD * 4
        banner   = _row_info_banner(row_info, canvas_w)
        row_img  = Image.new("RGB", (canvas_w, ROW_INFO_H + PANEL_H + LABEL_H), BG)
        row_img.paste(banner, (0, 0))
        for i, panel in enumerate([p1, p2, p3, p4, p5]):
            row_img.paste(panel, (i * (PANEL_W + PAD), ROW_INFO_H))
        rows.append(row_img)

    if not rows:
        print("No samples rendered.")
        return

    row_h    = rows[0].height
    canvas_w = rows[0].width
    legend   = _legend_strip(canvas_w)
    total_h  = row_h * len(rows) + PAD * (len(rows) - 1) + PAD + legend.height
    canvas   = Image.new("RGB", (canvas_w, total_h), BG)
    for i, row in enumerate(rows):
        canvas.paste(row, (0, i * (row_h + PAD)))
    canvas.paste(legend, (0, row_h * len(rows) + PAD * len(rows)))

    out_path = os.path.abspath(args.out)
    canvas.save(out_path)
    print(f"Saved: {out_path}  ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    main()
