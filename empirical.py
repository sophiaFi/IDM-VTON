"""Measure shoulder-hip distance for all person images in data/empirical.

Distance is taken between the midpoint of the two shoulder keypoints and the
midpoint of the two hip keypoints (OpenPose), reported in pixels of the
original image and as a fraction of image height.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

# OpenPose 18-keypoint indices
R_SHOULDER, L_SHOULDER, R_HIP, L_HIP = 2, 5, 8, 11
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def shoulder_hip_distance(keypoints, scale_x, scale_y):
    kp = np.array(keypoints["pose_keypoints_2d"], dtype=np.float64)
    idx = [R_SHOULDER, L_SHOULDER, R_HIP, L_HIP]
    if np.any(np.all(kp[idx] == 0, axis=1)):
        return None
    kp = kp * np.array([scale_x, scale_y])
    shoulder_c = kp[[R_SHOULDER, L_SHOULDER]].mean(axis=0)
    hip_c = kp[[R_HIP, L_HIP]].mean(axis=0)
    return float(np.linalg.norm(shoulder_c - hip_c)), kp, shoulder_c, hip_c


def body_extent(parsing_model, img):
    """Vertical/horizontal extent of the person (all non-background parsing labels), in img pixels."""
    w, h = img.size
    parsed, _ = parsing_model(img.resize((384, 512)))
    mask = np.array(parsed.resize((w, h), Image.NEAREST)) > 0
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    if len(rows) == 0:
        return None
    # +1 so a single-row extent still has height 1
    return float(rows[0]), float(rows[-1] + 1), float(cols[0])


def draw_measurement(img, kp, shoulder_c, hip_c, dist, body):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    r = max(3, w // 150)
    lw = max(2, w // 250)

    y_top, y_bot, x_min = body
    yellow = (255, 215, 0)
    x_line = max(2 * r, x_min - 6 * r)
    for y in (y_top, y_bot):
        draw.line([(0, y), (w, y)], fill=yellow, width=1)
        draw.line([(x_line - 2 * r, y), (x_line + 2 * r, y)], fill=yellow, width=lw)
    draw.line([(x_line, y_top), (x_line, y_bot)], fill=yellow, width=lw)

    for a, b in ((R_SHOULDER, L_SHOULDER), (R_HIP, L_HIP)):
        draw.line([tuple(kp[a]), tuple(kp[b])], fill=(0, 200, 255), width=lw)
    for i in (R_SHOULDER, L_SHOULDER, R_HIP, L_HIP):
        x, y = kp[i]
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(0, 200, 255))

    draw.line([tuple(shoulder_c), tuple(hip_c)], fill=(255, 40, 40), width=lw + 1)
    for x, y in (shoulder_c, hip_c):
        draw.ellipse([x - r - 1, y - r - 1, x + r + 1, y + r + 1], fill=(255, 40, 40))

    try:
        font = ImageFont.load_default(size=max(14, w // 40))
    except TypeError:  # Pillow < 10.1 has no sized default font
        font = ImageFont.load_default()

    def put_label(text, x, y):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        x = min(max(0, x), w - (right - left) - 4)
        y = min(max(0, y), h - (bottom - top) - 4)
        draw.rectangle([x, y, x + right - left + 4, y + bottom - top + 4], fill=(0, 0, 0))
        draw.text((x + 2 - left, y + 2 - top), text, fill=(255, 255, 255), font=font)

    mx, my = (shoulder_c + hip_c) / 2
    put_label(f"shoulder-hip: {dist:.1f} px ({dist / h:.3f} of image height)", mx + 3 * r, my)
    body_h = y_bot - y_top
    put_label(f"body height: {body_h:.1f} px ({body_h / h:.3f} of image height)", x_line, y_top - 2 * r - 30)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/empirical")
    parser.add_argument("--output", default="data/empirical/shoulder_hip_distance.json")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--visualize_dir",
        nargs="?",
        const="data/empirical/visualized",
        default=None,
        help="If set, save images with keypoints and the measured distance drawn on them "
        "(default dir when given without a value: data/empirical/visualized).",
    )
    args = parser.parse_args()
    if args.visualize_dir:
        Path(args.visualize_dir).mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    paths = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    openpose = OpenPose(args.gpu_id)
    parsing = Parsing(args.gpu_id)

    results, skipped = {}, []
    for path in paths:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        # OpenPose wrapper works at 384x512 and returns keypoints in that frame
        keypoints = openpose(img.resize((384, 512)))
        measured = shoulder_hip_distance(keypoints, w / 384, h / 512)
        if measured is None:
            print(f"[WARN] missing shoulder/hip keypoints: {path.name}")
            skipped.append(path.name)
            continue
        dist, kp, shoulder_c, hip_c = measured
        body = body_extent(parsing, img)
        if body is None:
            print(f"[WARN] no person found by parsing model: {path.name}")
            skipped.append(path.name)
            continue
        body_h = body[1] - body[0]
        if args.visualize_dir:
            vis = draw_measurement(img, kp, shoulder_c, hip_c, dist, body)
            vis.save(Path(args.visualize_dir) / f"{path.stem}.png")
        results[path.name] = {
            "distance_px": dist,
            "distance_rel_height": dist / h,
            "body_height_px": body_h,
            "image_size": [w, h],
        }
        print(f"{path.name}: shoulder-hip {dist:.1f} px ({dist / h:.3f} of image height), "
              f"body height {body_h:.1f} px")

    dists = np.array([r["distance_px"] for r in results.values()])
    rels = np.array([r["distance_rel_height"] for r in results.values()])
    body_hs = np.array([r["body_height_px"] for r in results.values()])
    summary = {
        "n_measured": len(results),
        "skipped": skipped,
        "mean_px": float(dists.mean()) if len(dists) else None,
        "std_px": float(dists.std()) if len(dists) else None,
        "mean_rel_height": float(rels.mean()) if len(rels) else None,
        "mean_body_height_px": float(body_hs.mean()) if len(body_hs) else None,
        "measurements": results,
    }
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMean shoulder-hip: {summary['mean_px']:.1f} px ({summary['mean_rel_height']:.3f} of image height), "
          f"mean body height: {summary['mean_body_height_px']:.1f} px, "
          f"n={len(results)} -> {args.output}")


if __name__ == "__main__":
    main()
