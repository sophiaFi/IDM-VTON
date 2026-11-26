"""Draw the assumed bust line on the person images in data/empirical for manual checking.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from empirical import IMG_EXTS, L_HIP, L_SHOULDER, R_HIP, R_SHOULDER, shoulder_hip_distance
from preprocess.openpose.run_openpose import OpenPose


def draw_bust_line(img, kp, bust_fraction):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    r = max(3, w // 150)
    lw = max(2, w // 250)
    cyan, green, red = (0, 200, 255), (0, 200, 0), (255, 40, 40)

    sides = ((R_SHOULDER, R_HIP), (L_SHOULDER, L_HIP))
    bust_pts = []
    for s, hp in sides:
        draw.line([tuple(kp[s]), tuple(kp[hp])], fill=cyan, width=lw)
        for i in (s, hp):
            x, y = kp[i]
            draw.ellipse([x - r, y - r, x + r, y + r], fill=cyan)
        bust_pts.append(kp[s] + bust_fraction * (kp[hp] - kp[s]))

    bust_y = float(np.mean([p[1] for p in bust_pts]))
    draw.line([(0, bust_y), (w, bust_y)], fill=red, width=lw)
    for x, y in bust_pts:
        draw.ellipse([x - r - 1, y - r - 1, x + r + 1, y + r + 1], fill=green, outline=(0, 0, 0))

    try:
        font = ImageFont.load_default(size=max(14, w // 40))
    except TypeError:  # Pillow < 10.1 has no sized default font
        font = ImageFont.load_default()
    label = f"bust line at {bust_fraction:.0%} (y={bust_y:.0f})"
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    x, y = 4, max(0, bust_y - (bottom - top) - 10)
    draw.rectangle([x, y, x + right - left + 4, y + bottom - top + 4], fill=(0, 0, 0))
    draw.text((x + 2 - left, y + 2 - top), label, fill=(255, 255, 255), font=font)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/empirical")
    parser.add_argument("--output_dir", default="data/empirical/bust_check")
    parser.add_argument("--bust_fraction", type=float, default=0.25)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in Path(args.data_dir).iterdir() if p.suffix.lower() in IMG_EXTS)
    openpose = OpenPose(args.gpu_id)

    for path in paths:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        keypoints = openpose(img.resize((384, 512)))
        measured = shoulder_hip_distance(keypoints, w / 384, h / 512)
        if measured is None:
            print(f"[WARN] missing shoulder/hip keypoints: {path.name}")
            continue
        draw_bust_line(img, measured[1], args.bust_fraction).save(out_dir / f"{path.stem}.png")
        print(f"saved {path.name}")


if __name__ == "__main__":
    main()
