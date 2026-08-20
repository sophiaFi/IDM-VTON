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


def get_mask_location(model_type, category, model_parse: Image.Image, keypoint: dict, width=384, height=512,
                      body_bust_cm: float = None, garment_bust_cm: float = None, densepose_torso_mask: np.ndarray | None = None):
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

    # Bust-driven horizontal expansion (upper_body / dresses only)
    if (category in ('upper_body', 'dresses')
            and body_bust_cm is not None
            and garment_bust_cm is not None):

        bust_ratio = garment_bust_cm / max(body_bust_cm, 1e-6)

        if bust_ratio > 1.0:
            # Body bust width in pixels from keypoints
            pose_data_bust = np.array(keypoint["pose_keypoints_2d"]).reshape(-1, 2).astype(np.float32)
            bust_fraction = 0.25
            left_shoulder  = pose_data_bust[2, :2] * (height / 512.0)
            right_shoulder = pose_data_bust[5, :2] * (height / 512.0)
            left_hip       = pose_data_bust[8, :2] * (height / 512.0)
            right_hip      = pose_data_bust[11, :2] * (height / 512.0)
            left_bust_pt   = left_shoulder  + bust_fraction * (left_hip  - left_shoulder)
            right_bust_pt  = right_shoulder + bust_fraction * (right_hip - right_shoulder)
            body_bust_px   = float(np.linalg.norm(right_bust_pt - left_bust_pt))

            # Target mask bust width in pixels
            target_mask_bust_px = body_bust_px * bust_ratio

            print(f"target_mask_bust_px: {target_mask_bust_px}")

            # Current torso bust width from the raw pre-dilation mask,
            # clipped to the DensePose torso column range to exclude sleeves
            # Falls back to full mask width if no torso mask is available
            bust_y     = int(round((left_bust_pt[1] + right_bust_pt[1]) / 2.0))

            torso_np = None
            if densepose_torso_mask is not None:
                torso_np = np.array(densepose_torso_mask)
                if torso_np.shape != parse_mask_raw.shape:
                    torso_np = cv2.resize(torso_np, (width, height),
                                          interpolation=cv2.INTER_NEAREST)

            torso_bounds = _torso_x_bounds_at_row(torso_np, bust_y) if torso_np is not None else None

            if torso_bounds is not None:
                clip_left, clip_right = torso_bounds
            else:
                clip_left, clip_right = 0, width - 1

            current_mask_bust_px = _mask_bust_width_px_clipped(
                parse_mask_raw, bust_y, clip_left, clip_right,
            )

            print(f"current_mask_bust_px: {current_mask_bust_px}")

            # Expand horizontally if the torso region is narrower than needed
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
