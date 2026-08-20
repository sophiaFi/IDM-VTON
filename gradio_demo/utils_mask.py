import os
import numpy as np
import cv2
from PIL import Image, ImageDraw

label_map = {
    "background": 0,
    "hat": 1,
    "hair": 2,
    "sunglasses": 3,
    "upper_clothes": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "left_shoe": 9,
    "right_shoe": 10,
    "head": 11,
    "left_leg": 12,
    "right_leg": 13,
    "left_arm": 14,
    "right_arm": 15,
    "bag": 16,
    "scarf": 17,
}

def extend_arm_mask(wrist, elbow, scale):
  wrist = elbow + scale * (wrist - elbow)
  return wrist

def hole_fill(img):
    img = np.pad(img[1:-1, 1:-1], pad_width = 1, mode = 'constant', constant_values=0)
    img_copy = img.copy()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)

    cv2.floodFill(img, mask, (0, 0), 255)
    img_inverse = cv2.bitwise_not(img)
    dst = cv2.bitwise_or(img_copy, img_inverse)
    return dst

def refine_mask(mask):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    area = []
    for j in range(len(contours)):
        a_d = cv2.contourArea(contours[j], True)
        area.append(abs(a_d))
    refine_mask = np.zeros_like(mask).astype(np.uint8)
    if len(area) != 0:
        i = area.index(max(area))
        cv2.drawContours(refine_mask, contours, i, color=255, thickness=-1)

    return refine_mask

# Maximum bust expansion added per side in pixels (at 384×512).
# Caps runaway expansion caused by noisy keypoints or an atypical bust ratio.
MAX_BUST_EXPANSION_PX = 40

# Maximum downward expansion in pixels (at 384×512).
MAX_LENGTH_EXPANSION_PX = 100

# Shoulder-to-hip vertical span as a fraction of total body height.
# Acromion to greater trochanter ≈ 30% of stature (low population variance).
SHOULDER_HIP_TO_HEIGHT_RATIO = 0.30


def _torso_x_bounds_at_row(
    torso_mask_np: np.ndarray,
    bust_y: int,
    band: int = 5,
    padding: int = 3,
) -> tuple | None:
    """Return (left_x, right_x) of the DensePose torso region at bust_y.

    Scans a band of rows and returns the widest torso span found.
    The resulting bounds are expanded by `padding` pixels on both sides.
    Returns None if no torso pixels are found in the band.
    """
    h, w = torso_mask_np.shape
    y0 = max(0, bust_y - band)
    y1 = min(h, bust_y + band + 1)
    best_left, best_right, best_width = 0, 0, 0
    for y in range(y0, y1):
        cols = np.where(torso_mask_np[y] > 127)[0]
        if len(cols) == 0:
            continue
        span = int(cols[-1]) - int(cols[0]) + 1
        if span > best_width:
            best_width = span
            best_left  = int(cols[0])
            best_right = int(cols[-1])

    if best_width <= 0:
        return None

    # add padding
    best_left = max(0, best_left - padding)
    best_right = min(w - 1, best_right + padding)

    return best_left, best_right


def _mask_bust_width_px_clipped(
    mask_np: np.ndarray,
    bust_y: int,
    clip_left: int,
    clip_right: int,
    band: int = 5,
) -> float:
    """Return the width of the garment mask at bust_y, restricted to [clip_left, clip_right].

    Counts all white pixels in that column range across the band and returns
    the span from the leftmost to rightmost hit. Returns 0.0 if none found.
    """
    h, w = mask_np.shape
    y0 = max(0, bust_y - band)
    y1 = min(h, bust_y + band + 1)
    clip_l = max(0, clip_left)
    clip_r = min(w - 1, clip_right)
    best_left, best_right, found = clip_r, clip_l, False
    for y in range(y0, y1):
        row = mask_np[y, clip_l:clip_r + 1] > 127
        cols = np.where(row)[0]
        if len(cols) == 0:
            continue
        found = True
        best_left  = min(best_left,  clip_l + int(cols[0]))
        best_right = max(best_right, clip_l + int(cols[-1]))
    return float(best_right - best_left + 1) if found else 0.0


def _garment_vertical_extent_px(
    mask_np: np.ndarray,
    torso_bounds: tuple | None,
    midpoint_x: int,
    shoulder_y: int,
    fallback_band_pct: float = 0.10,
) -> tuple[int, int] | None:
    """Return (top_y, hem_y) of the garment in the torso column corridor.

    top_y is found via connected components: the component touching the shoulder
    keypoint row is followed to its topmost row, anchoring measurement at the
    shoulder seam rather than the collar. hem_y is the bottommost row in the
    corridor that contains any white mask pixel.

    Falls back to the raw shoulder_y if the keypoint falls outside the mask.
    Returns None if no mask pixels are found in the corridor.
    """
    h, w = mask_np.shape
    if torso_bounds is not None:
        clip_left, clip_right = torso_bounds
    else:
        band = max(1, int(w * fallback_band_pct))
        clip_left  = max(0, midpoint_x - band)
        clip_right = min(w - 1, midpoint_x + band)

    # Binary mask restricted to torso columns
    region_full = np.zeros_like(mask_np, dtype=np.uint8)
    region_full[:, clip_left : clip_right + 1] = (mask_np[:, clip_left : clip_right + 1] > 127).astype(np.uint8) * 255

    if (region_full > 127).sum() == 0:
        return None

    # Connected component anchored at shoulder keypoint for both top and hem
    _, labels = cv2.connectedComponents(region_full, connectivity=8)
    anchor_y = max(0, min(h - 1, shoulder_y))
    band_rows = range(max(0, anchor_y - 5), min(h, anchor_y + 6))
    anchor_labels = [
        labels[r, c]
        for r in band_rows
        for c in range(clip_left, clip_right + 1)
        if labels[r, c] != 0
    ]
    # hem: bottommost white pixel on a narrow vertical band through the shoulder midpoint,
    # so that a sleeve entering the corridor from the side does not affect the result
    centre_band = 5
    centre_l = max(clip_left,  midpoint_x - centre_band)
    centre_r = min(clip_right, midpoint_x + centre_band)
    centre_rows = np.where((region_full[:, centre_l:centre_r + 1] > 127).any(axis=1))[0]
    hem_y = int(centre_rows[-1]) if len(centre_rows) > 0 else None

    if anchor_labels:
        garment_label = max(set(anchor_labels), key=anchor_labels.count)
        component_rows = np.where((labels == garment_label).any(axis=1))[0]
        top_y = int(component_rows[0])
    else:
        # Shoulder keypoint is outside the mask — use keypoint row directly
        top_y = anchor_y

    if hem_y is None:
        # Centre column has no mask pixels — fall back to the full corridor bottom
        all_rows = np.where((region_full > 127).any(axis=1))[0]
        hem_y = int(all_rows[-1])

    return top_y, hem_y


def get_mask_location(model_type, category, model_parse: Image.Image, keypoint: dict, width=384, height=512,
                      body_bust_cm: float = None, garment_bust_cm: float = None,
                      body_hips_cm: float = None, garment_length_cm: float = None, body_height_cm: float = None,
                      densepose_torso_mask: np.ndarray | None = None,
                      debug_dir: str = None):
    im_parse = model_parse.resize((width, height), Image.NEAREST)
    parse_array = np.array(im_parse)

    if model_type == 'hd':
        arm_width = 60
    elif model_type == 'dc':
        arm_width = 45
    else:
        raise ValueError("model_type must be \'hd\' or \'dc\'!")

    parse_head = (parse_array == 1).astype(np.float32) + \
                 (parse_array == 3).astype(np.float32) + \
                 (parse_array == 11).astype(np.float32)

    parser_mask_fixed = (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["hat"]).astype(np.float32) + \
                        (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                        (parse_array == label_map["bag"]).astype(np.float32)

    parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)

    arms_left = (parse_array == 14).astype(np.float32)
    arms_right = (parse_array == 15).astype(np.float32)

    if category == 'dresses':
        parse_mask = (parse_array == 7).astype(np.float32) + \
                     (parse_array == 4).astype(np.float32) + \
                     (parse_array == 5).astype(np.float32) + \
                     (parse_array == 6).astype(np.float32)

        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

    elif category == 'upper_body':
        parse_mask = (parse_array == 4).astype(np.float32) + (parse_array == 7).astype(np.float32)
        parser_mask_fixed_lower_cloth = (parse_array == label_map["skirt"]).astype(np.float32) + \
                                        (parse_array == label_map["pants"]).astype(np.float32)
        # When length expansion is active the garment may extend over the leg area,
        # so lower clothing must be changeable rather than fixed.
        if garment_length_cm is None or body_height_cm is None:
            parser_mask_fixed += parser_mask_fixed_lower_cloth
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
    elif category == 'lower_body':
        parse_mask = (parse_array == 6).astype(np.float32) + \
                     (parse_array == 12).astype(np.float32) + \
                     (parse_array == 13).astype(np.float32) + \
                     (parse_array == 5).astype(np.float32)
        parser_mask_fixed += (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
                             (parse_array == 14).astype(np.float32) + \
                             (parse_array == 15).astype(np.float32)
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
    else:
        raise NotImplementedError

    # Load pose points
    pose_data = keypoint["pose_keypoints_2d"]
    pose_data = np.array(pose_data)
    pose_data = pose_data.reshape((-1, 2))

    im_arms_left = Image.new('L', (width, height))
    im_arms_right = Image.new('L', (width, height))
    arms_draw_left = ImageDraw.Draw(im_arms_left)
    arms_draw_right = ImageDraw.Draw(im_arms_right)
    if category == 'dresses' or category == 'upper_body':
        shoulder_right = np.multiply(tuple(pose_data[2][:2]), height / 512.0)
        shoulder_left = np.multiply(tuple(pose_data[5][:2]), height / 512.0)
        elbow_right = np.multiply(tuple(pose_data[3][:2]), height / 512.0)
        elbow_left = np.multiply(tuple(pose_data[6][:2]), height / 512.0)
        wrist_right = np.multiply(tuple(pose_data[4][:2]), height / 512.0)
        wrist_left = np.multiply(tuple(pose_data[7][:2]), height / 512.0)
        ARM_LINE_WIDTH = int(arm_width / 512 * height)
        size_left = [shoulder_left[0] - ARM_LINE_WIDTH // 2, shoulder_left[1] - ARM_LINE_WIDTH // 2, shoulder_left[0] + ARM_LINE_WIDTH // 2, shoulder_left[1] + ARM_LINE_WIDTH // 2]
        size_right = [shoulder_right[0] - ARM_LINE_WIDTH // 2, shoulder_right[1] - ARM_LINE_WIDTH // 2, shoulder_right[0] + ARM_LINE_WIDTH // 2,
                      shoulder_right[1] + ARM_LINE_WIDTH // 2]
        

        if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
            im_arms_right = arms_right
        else:
            wrist_right = extend_arm_mask(wrist_right, elbow_right, 1.2)
            arms_draw_right.line(np.concatenate((shoulder_right, elbow_right, wrist_right)).astype(np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
            arms_draw_right.arc(size_right, 0, 360, 'white', ARM_LINE_WIDTH // 2)

        if wrist_left[0] <= 1. and wrist_left[1] <= 1.:
            im_arms_left = arms_left
        else:
            wrist_left = extend_arm_mask(wrist_left, elbow_left, 1.2)
            arms_draw_left.line(np.concatenate((wrist_left, elbow_left, shoulder_left)).astype(np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
            arms_draw_left.arc(size_left, 0, 360, 'white', ARM_LINE_WIDTH // 2)

        hands_left = np.logical_and(np.logical_not(im_arms_left), arms_left)
        hands_right = np.logical_and(np.logical_not(im_arms_right), arms_right)
        parser_mask_fixed += hands_left + hands_right

    parser_mask_fixed = np.logical_or(parser_mask_fixed, parse_head)

    # Snapshot the raw mask first so step 3 measures the undilated garment region
    parse_mask_raw = (parse_mask > 0.5).astype(np.uint8) * 255
    # Small base dilation to smooth the garment boundary
    parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16), iterations=2)

    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        Image.fromarray((parse_mask > 0.5).astype(np.uint8) * 255).save(
            os.path.join(debug_dir, "mask_before_expansion.png")
        )
        Image.fromarray((parse_mask_raw > 0.5).astype(np.uint8) * 255).save(
            os.path.join(debug_dir, "raw_mask_before_expansion.png")
        )

    # Shared keypoint geometry used by both expansion steps
    _do_expansion = (category in ('upper_body', 'dresses') and (
        (body_bust_cm is not None and garment_bust_cm is not None) or
        (garment_length_cm is not None and body_height_cm is not None)
    ))

    if _do_expansion:
        pose_data_kp = np.array(keypoint["pose_keypoints_2d"]).reshape(-1, 2).astype(np.float32)
        scale = height / 512.0
        left_shoulder  = pose_data_kp[2,  :2] * scale
        right_shoulder = pose_data_kp[5,  :2] * scale
        left_hip       = pose_data_kp[8,  :2] * scale
        right_hip      = pose_data_kp[11, :2] * scale
        bust_fraction  = 0.25
        left_bust_pt   = left_shoulder + bust_fraction * (left_hip  - left_shoulder)
        right_bust_pt  = right_shoulder + bust_fraction * (right_hip - right_shoulder)
        bust_y         = int(round((left_bust_pt[1] + right_bust_pt[1]) / 2.0))
        midpoint_x     = int(round((left_bust_pt[0] + right_bust_pt[0]) / 2.0))

        # Prepare DensePose torso mask once (shared by both steps)
        torso_np = None
        if densepose_torso_mask is not None:
            torso_np = np.array(densepose_torso_mask)
            if torso_np.shape != parse_mask_raw.shape:
                torso_np = cv2.resize(torso_np, (width, height),
                                      interpolation=cv2.INTER_NEAREST)
        torso_bounds = _torso_x_bounds_at_row(torso_np, bust_y) if torso_np is not None else None

        # Bust-driven horizontal expansion
        if body_bust_cm is not None and garment_bust_cm is not None:
            bust_ratio = garment_bust_cm / max(body_bust_cm, 1e-6)
            if bust_ratio > 1.0:
                body_bust_px = float(np.linalg.norm(right_bust_pt - left_bust_pt))
                target_mask_bust_px = body_bust_px * bust_ratio
                print(f"target_mask_bust_px: {target_mask_bust_px}")

                if torso_bounds is not None:
                    clip_left, clip_right = torso_bounds
                else:
                    clip_left, clip_right = 0, width - 1

                current_mask_bust_px = _mask_bust_width_px_clipped(
                    parse_mask_raw, bust_y, clip_left, clip_right,
                )
                print(f"current_mask_bust_px: {current_mask_bust_px}")

                if current_mask_bust_px > 0 and current_mask_bust_px < target_mask_bust_px:
                    extra_each_side = int(round((target_mask_bust_px - current_mask_bust_px) / 2.0))
                    extra_each_side = min(extra_each_side, MAX_BUST_EXPANSION_PX)
                    print(f"extra_each_side: {extra_each_side}")
                    if extra_each_side >= 1:
                        kernel_w = 2 * extra_each_side + 1
                        parse_mask = cv2.dilate(
                            parse_mask.astype(np.uint8),
                            np.ones((1, kernel_w), np.uint8),
                        )

        # Length-driven downward expansion
        if garment_length_cm is not None and body_height_cm is not None:
            # px/cm scale from shoulder-to-hip keypoint distance vs. known body height
            shoulder_hip_px = float(np.linalg.norm(
                ((left_shoulder + right_shoulder) - (left_hip + right_hip)) / 2.0
            ))
            px_per_cm = shoulder_hip_px / max(SHOULDER_HIP_TO_HEIGHT_RATIO * body_height_cm, 1e-6)
            print(f"px_per_cm: {px_per_cm:.3f}")

            body_height_px = body_height_cm * px_per_cm

            shoulder_y = int(round((left_shoulder[1] + right_shoulder[1]) / 2.0))
            extent = _garment_vertical_extent_px(
                parse_mask_raw, torso_bounds, midpoint_x, shoulder_y,
            )

            if extent is not None:
                garment_top_px, garment_hem_px = extent
                current_garment_length_px = garment_hem_px - garment_top_px
                target_garment_length_px  = (garment_length_cm / max(body_height_cm, 1e-6)) * body_height_px
                print(f"current_garment_length_px: {current_garment_length_px:.1f}, "
                      f"target_garment_length_px: {target_garment_length_px:.1f}")

                if current_garment_length_px < target_garment_length_px:
                    extra_down = int(round(target_garment_length_px - current_garment_length_px))
                    extra_down = min(extra_down, MAX_LENGTH_EXPANSION_PX)
                    print(f"extra_down: {extra_down}")
                    if extra_down >= 1:
                        # Asymmetric kernel: anchor at bottom row → expands downward only
                        kernel_h = extra_down + 1
                        kernel = np.ones((kernel_h, 1), np.uint8)
                        parse_mask = cv2.dilate(
                            parse_mask.astype(np.uint8),
                            kernel,
                            anchor=(0, kernel_h - 1),
                        )

                if debug_dir is not None:
                    # Visualize important points and lines used for calculations

                    # Column corridor used by _garment_vertical_extent_px
                    if torso_bounds is not None:
                        viz_l, viz_r = torso_bounds
                    else:
                        _band = max(1, int(width * 0.10))
                        viz_l = max(0, midpoint_x - _band)
                        viz_r = min(width - 1, midpoint_x + _band)

                    # Base: raw garment mask as grayscale → RGB
                    canvas = np.stack([parse_mask_raw] * 3, axis=-1).astype(np.uint8)
                    # Tint non-garment pixels inside the corridor dark purple so the
                    # search region is visible even where the mask is absent
                    corridor = canvas[:, viz_l:viz_r + 1]
                    no_garment = corridor[:, :, 0] == 0
                    corridor[no_garment] = [40, 20, 50]
                    canvas[:, viz_l:viz_r + 1] = corridor

                    # Shoulder anchor band (±5 rows) — yellow additive tint
                    anchor_y = max(0, min(height - 1, shoulder_y))
                    b0 = max(0, anchor_y - 5)
                    b1 = min(height - 1, anchor_y + 5)
                    band = canvas[b0:b1 + 1, viz_l:viz_r + 1].astype(np.int32)
                    canvas[b0:b1 + 1, viz_l:viz_r + 1] = np.clip(band + [50, 50, 0], 0, 255).astype(np.uint8)

                    def _dline(arr, y, x0, x1, color, dash=5, gap=4, thickness=2):
                        y = int(round(max(0, min(arr.shape[0] - thickness, y))))
                        x = x0
                        while x <= x1:
                            xe = min(x + dash, x1 + 1)
                            arr[y:y + thickness, x:xe] = color
                            x += dash + gap

                    def _sline(arr, y, x0, x1, color, thickness=2):
                        y = int(round(max(0, min(arr.shape[0] - thickness, y))))
                        arr[y:y + thickness, x0:x1 + 1] = color

                    # Torso corridor bounds — magenta vertical lines
                    canvas[:, viz_l:viz_l + 2]  = [220, 0, 220]
                    canvas[:, viz_r - 1:viz_r + 1] = [220, 0, 220]
                    # Shoulder row — white dashed
                    _dline(canvas, shoulder_y, viz_l, viz_r, [255, 255, 255])
                    # Garment top (connected-component) — light blue solid
                    _sline(canvas, garment_top_px, viz_l, viz_r, [130, 200, 255])
                    # Current hem — dark blue solid
                    _sline(canvas, garment_hem_px, viz_l, viz_r, [30, 80, 220])
                    # Target hem — yellow dashed
                    target_y_viz = int(round(min(garment_top_px + target_garment_length_px, height - 1)))
                    _dline(canvas, target_y_viz, viz_l, viz_r, [255, 220, 0])

                    Image.fromarray(canvas).save(os.path.join(debug_dir, "extent_debug.png"))

    if debug_dir is not None:
        Image.fromarray((parse_mask > 0.5).astype(np.uint8) * 255).save(
            os.path.join(debug_dir, "mask_after_expansion.png")
        )

    if category == 'dresses' or category == 'upper_body':
        neck_mask = (parse_array == 18).astype(np.float32)
        neck_mask = cv2.dilate(neck_mask, np.ones((5, 5), np.uint16), iterations=1)
        neck_mask = np.logical_and(neck_mask, np.logical_not(parse_head))
        parse_mask = np.logical_or(parse_mask, neck_mask)
        arm_mask = cv2.dilate(np.logical_or(im_arms_left, im_arms_right).astype('float32'), np.ones((5, 5), np.uint16), iterations=4)
        parse_mask += np.logical_or(parse_mask, arm_mask)

    parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))

    parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
    inpaint_mask = 1 - parse_mask_total
    img = np.where(inpaint_mask, 255, 0)
    dst = hole_fill(img.astype(np.uint8))
    dst = refine_mask(dst)
    inpaint_mask = dst / 255 * 1
    mask = Image.fromarray(inpaint_mask.astype(np.uint8) * 255)
    mask_gray = Image.fromarray(inpaint_mask.astype(np.uint8) * 127)

    return mask, mask_gray



# Helper functions for visualization

def get_bust_measurement_lines(
    keypoint: dict,
    mask: Image.Image,
    densepose_torso_mask: np.ndarray | None = None,
    height: int = 512,
    bust_fraction: float = 0.25,
    band: int = 5,
) -> dict:
    """Return the two bust measurement lines used during mask expansion.

    Both lines are horizontal and share the same y-coordinate (bust_y).

    When densepose_torso_mask is provided (a uint8 HxW array, 255 = torso),
    the mask measurement is clipped to the DensePose torso column range so
    sleeve pixels are excluded. Without it, the full mask width is returned.

    Returns a dict with:
      bust_y            — the row at which bust lines are drawn
      body_line         — ((x0, y), (x1, y)) spanning body_bust_px, centred on midpoint_x
      mask_line         — ((x0, y), (x1, y)) spanning the garment mask within the
                          torso corridor, or None if no mask pixels found there
      torso_bounds      — (left_x, right_x) from DensePose, or None
      left_shoulder     — (x, y)
      left_hip          — (x, y)
      right_shoulder    — (x, y)
      right_hip         — (x, y)
    """
    pose_data = np.array(keypoint["pose_keypoints_2d"]).reshape(-1, 2).astype(np.float32)
    scale = height / 512.0
    left_shoulder  = pose_data[2, :2] * scale
    right_shoulder = pose_data[5, :2] * scale
    left_hip       = pose_data[8, :2] * scale
    right_hip      = pose_data[11, :2] * scale
    left_bust_pt   = left_shoulder  + bust_fraction * (left_hip  - left_shoulder)
    right_bust_pt  = right_shoulder + bust_fraction * (right_hip - right_shoulder)

    body_bust_px = float(np.linalg.norm(right_bust_pt - left_bust_pt))
    bust_y       = int(round((left_bust_pt[1] + right_bust_pt[1]) / 2.0))
    midpoint_x   = int(round((left_bust_pt[0] + right_bust_pt[0]) / 2.0))

    half_body = body_bust_px / 2.0
    body_line = (
        (midpoint_x - half_body, bust_y),
        (midpoint_x + half_body, bust_y),
    )

    mask_np  = np.array(mask.convert("L"))
    mask_raw = (mask_np > 127).astype(np.uint8) * 255
    w = mask_raw.shape[1]

    torso_bounds = None
    if densepose_torso_mask is not None:
        torso_np = np.array(densepose_torso_mask)
        if torso_np.shape != mask_raw.shape:
            torso_np = cv2.resize(torso_np, (mask_raw.shape[1], mask_raw.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        torso_bounds = _torso_x_bounds_at_row(torso_np, bust_y, band)

    if torso_bounds is not None:
        clip_left, clip_right = torso_bounds
        mask_width = _mask_bust_width_px_clipped(mask_raw, bust_y, clip_left, clip_right, band)
        mask_midpoint = (clip_left + clip_right) / 2.0
        mask_line = (
            (mask_midpoint - mask_width / 2.0, float(bust_y)),
            (mask_midpoint + mask_width / 2.0, float(bust_y)),
        ) if mask_width > 0 else None
    else:
        # Fallback: full mask width centred on the keypoint midpoint
        mask_width = _mask_bust_width_px_clipped(mask_raw, bust_y, 0, w - 1, band)
        mask_line = (
            (midpoint_x - mask_width / 2.0, float(bust_y)),
            (midpoint_x + mask_width / 2.0, float(bust_y)),
        ) if mask_width > 0 else None

    return {
        "bust_y":          bust_y,
        "body_line":       body_line,
        "mask_line":       mask_line,
        "torso_bounds":    torso_bounds,
        "left_shoulder":   tuple(left_shoulder.tolist()),
        "left_hip":        tuple(left_hip.tolist()),
        "right_shoulder":  tuple(right_shoulder.tolist()),
        "right_hip":       tuple(right_hip.tolist()),
    }


def get_length_measurement_lines(
    keypoint: dict,
    model_parse: Image.Image,
    category: str,
    garment_length_cm: float,
    body_height_cm: float,
    width: int = 384,
    densepose_torso_mask: np.ndarray | None = None,
    height: int = 512,
) -> dict:
    """Return measurement geometry for the length expansion visualisation.

    Uses the raw garment segmentation (same labels as get_mask_location) so that
    the vertical extent matches what the expansion logic actually measures.

    Returns a dict with:
      torso_bounds          — (left_x, right_x) column corridor, or None
      shoulder_y            — keypoint-averaged shoulder row
      garment_top_y         — topmost row of the garment component touching shoulder
      garment_hem_y         — current bottommost row of garment in corridor
      target_garment_hem_y  — where the hem should be after expansion
      shoulder_hip_line     — ((left_shoulder_pt, left_hip_pt), (right_shoulder_pt, right_hip_pt))
                              the two vertical reference segments used to derive px/cm
      px_per_cm             — derived scale factor
      left_shoulder         — (x, y)
      right_shoulder        — (x, y)
      left_hip              — (x, y)
      right_hip             — (x, y)
    """
    # Extract the same raw garment mask used inside get_mask_location
    im_parse    = model_parse.resize((width, height), Image.NEAREST)
    parse_array = np.array(im_parse)
    if category == 'upper_body':
        parse_mask = (parse_array == 4).astype(np.float32) + (parse_array == 7).astype(np.float32)
    elif category == 'dresses':
        parse_mask = (
            (parse_array == 7).astype(np.float32) + (parse_array == 4).astype(np.float32) +
            (parse_array == 5).astype(np.float32) + (parse_array == 6).astype(np.float32)
        )
    else:
        raise ValueError(f"category '{category}' not supported for length measurement")
    mask_raw = (parse_mask > 0.5).astype(np.uint8) * 255

    pose_data  = np.array(keypoint["pose_keypoints_2d"]).reshape(-1, 2).astype(np.float32)
    scale      = height / 512.0
    left_shoulder  = pose_data[2,  :2] * scale
    right_shoulder = pose_data[5,  :2] * scale
    left_hip       = pose_data[8,  :2] * scale
    right_hip      = pose_data[11, :2] * scale

    bust_fraction = 0.25
    left_bust_pt  = left_shoulder  + bust_fraction * (left_hip  - left_shoulder)
    right_bust_pt = right_shoulder + bust_fraction * (right_hip - right_shoulder)
    bust_y        = int(round((left_bust_pt[1] + right_bust_pt[1]) / 2.0))
    midpoint_x    = int(round((left_bust_pt[0] + right_bust_pt[0]) / 2.0))
    shoulder_y    = int(round((left_shoulder[1] + right_shoulder[1]) / 2.0))

    torso_bounds = None
    if densepose_torso_mask is not None:
        torso_np = np.array(densepose_torso_mask)
        if torso_np.shape != mask_raw.shape:
            torso_np = cv2.resize(torso_np, (width, height), interpolation=cv2.INTER_NEAREST)
        torso_bounds = _torso_x_bounds_at_row(torso_np, bust_y)

    # px/cm from shoulder-to-hip keypoint distance vs. known body height
    shoulder_hip_px = float(np.linalg.norm(
        ((left_shoulder + right_shoulder) - (left_hip + right_hip)) / 2.0
    ))
    px_per_cm = shoulder_hip_px / max(SHOULDER_HIP_TO_HEIGHT_RATIO * body_height_cm, 1e-6)

    body_height_px = body_height_cm * px_per_cm

    extent = _garment_vertical_extent_px(mask_raw, torso_bounds, midpoint_x, shoulder_y)
    garment_top_y = extent[0] if extent is not None else shoulder_y
    garment_hem_y = extent[1] if extent is not None else shoulder_y

    target_length_px     = (garment_length_cm / max(body_height_cm, 1e-6)) * body_height_px
    target_garment_hem_y = garment_top_y + target_length_px

    return {
        "torso_bounds":         torso_bounds,
        "shoulder_y":           shoulder_y,
        "garment_top_y":        garment_top_y,
        "garment_hem_y":        garment_hem_y,
        "target_garment_hem_y": target_garment_hem_y,
        "shoulder_hip_line":    (
            (tuple(left_shoulder.tolist()),  tuple(left_hip.tolist())),
            (tuple(right_shoulder.tolist()), tuple(right_hip.tolist())),
        ),
        "px_per_cm":            px_per_cm,
        "left_shoulder":        tuple(left_shoulder.tolist()),
        "right_shoulder":       tuple(right_shoulder.tolist()),
        "left_hip":             tuple(left_hip.tolist()),
        "right_hip":            tuple(right_hip.tolist()),
    }
