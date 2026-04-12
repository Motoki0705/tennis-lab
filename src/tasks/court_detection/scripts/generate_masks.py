"""Pre-generate court cell segmentation masks from keypoint annotations.

For each annotated image, computes a homography from the 14 ground-plane
keypoints to the 3-D court model, projects the 6 in-court cell polygons
onto the image, and writes a single-channel ``uint8`` mask to
``data/court/masks/{id}.png``.

Label map
---------
0 : background
1 : left service box
2 : right service box
3 : left back court
4 : right back court
5 : left doubles alley
6 : right doubles alley

Usage::

    python -m src.tasks.court_detection.scripts.generate_masks --root data/court
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.utils.schema.court import (
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    HALF_SINGLES_WIDTH,
    SERVICE_LINE_DISTANCE,
    court_keypoints_3d,
)

_xs = HALF_SINGLES_WIDTH
_xd = HALF_DOUBLES_WIDTH
_ys = SERVICE_LINE_DISTANCE
_yB = HALF_LENGTH

# label → (x_min, x_max, y_min, y_max)
CELL_BOUNDS: dict[int, tuple[float, float, float, float]] = {
    1: (-_xs, 0.0,  0.0,  _ys),
    2: (0.0,  _xs,  0.0,  _ys),
    3: (-_xs, 0.0,  _ys,  _yB),
    4: (0.0,  _xs,  _ys,  _yB),
    5: (-_xd, -_xs, 0.0,  _yB),
    6: (_xs,  _xd,  0.0,  _yB),
}

LABEL_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (255, 80, 80),
    2: (80, 255, 80),
    3: (80, 80, 255),
    4: (255, 220, 80),
    5: (255, 80, 255),
    6: (80, 255, 255),
}


def _cell_corners(
    x_min: float, x_max: float, y_min: float, y_max: float,
) -> np.ndarray:
    return np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
    ], dtype=np.float32)


def _mirror_bounds_origin(
    x_min: float, x_max: float, y_min: float, y_max: float,
) -> tuple[float, float, float, float]:
    return -x_max, -x_min, -y_max, -y_min


def _compute_homography(kps_2d: np.ndarray) -> np.ndarray | None:
    kp3d = court_keypoints_3d()[:14].numpy()[:, :2]
    H, _status = cv2.findHomography(kp3d, kps_2d, cv2.RANSAC, 5.0)
    return H


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a label mask (0-6) to a color visualization image (BGR)."""
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label, bgr in LABEL_COLORS.items():
        color[mask == label] = bgr
    return color


def generate_mask(h: int, w: int, kps_2d: np.ndarray) -> np.ndarray | None:
    """Generate a court cell mask for one image."""
    H = _compute_homography(kps_2d)
    if H is None:
        return None

    mask = np.zeros((h, w), dtype=np.uint8)

    for label, (x_min, x_max, y_min, y_max) in CELL_BOUNDS.items():
        corners_3d = _cell_corners(x_min, x_max, y_min, y_max)
        corners_img = cv2.perspectiveTransform(
            corners_3d.reshape(1, -1, 2), H,
        ).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [corners_img], int(label))

        nx_min, nx_max, ny_min, ny_max = _mirror_bounds_origin(
            x_min, x_max, y_min, y_max,
        )
        corners_near = _cell_corners(nx_min, nx_max, ny_min, ny_max)
        corners_near_img = cv2.perspectiveTransform(
            corners_near.reshape(1, -1, 2), H,
        ).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [corners_near_img], int(label))

    return mask


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate court cell masks.")
    parser.add_argument("--root", default="data/court", help="Court dataset root.")
    parser.add_argument("--image-id", default=None, help="Process only one sample.")
    parser.add_argument("--write-color-mask", action="store_true")
    parser.add_argument("--color-dir-name", default="masks_color")
    args = parser.parse_args()

    root = Path(args.root)
    masks_dir = root / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    color_masks_dir = root / args.color_dir_name
    if args.write_color_mask:
        color_masks_dir.mkdir(parents=True, exist_ok=True)
    images_dir = root / "images"

    total = 0
    skipped = 0
    errors = 0

    for split in ("train", "val"):
        json_path = root / f"data_{split}.json"
        if not json_path.exists():
            print(f"[generate_masks] {json_path} not found, skipping.")
            continue
        with open(json_path) as f:
            entries = json.load(f)

        print(f"[generate_masks] Processing {split}: {len(entries)} samples")
        for entry in entries:
            if args.image_id is not None and entry["id"] != args.image_id:
                continue
            total += 1
            image_id = entry["id"]
            kps = np.array(entry["kps"], dtype=np.float32)

            img_path = images_dir / f"{image_id}.png"
            if not img_path.exists():
                img_path = images_dir / f"{image_id}.jpg"
            if not img_path.exists():
                print(f"  SKIP (no image): {image_id}")
                skipped += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  ERROR (imread): {image_id}")
                errors += 1
                continue

            h, w = img.shape[:2]
            mask = generate_mask(h, w, kps)
            if mask is None:
                print(f"  ERROR (homography): {image_id}")
                errors += 1
                continue

            out_path = masks_dir / f"{image_id}.png"
            cv2.imwrite(str(out_path), mask)
            if args.write_color_mask:
                color_mask = colorize_mask(mask)
                color_out_path = color_masks_dir / f"{image_id}.png"
                cv2.imwrite(str(color_out_path), color_mask)

            if args.image_id is not None:
                print(f"[generate_masks] Wrote single sample: {image_id}")
                return

    print(
        f"\n[generate_masks] Done: {total} total, "
        f"{total - skipped - errors} written, "
        f"{skipped} skipped, {errors} errors."
    )


if __name__ == "__main__":
    main()
