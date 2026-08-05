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

Usage:
    python -m src.tasks.court_detection.scripts.generate_masks
    python -m src.tasks.court_detection.scripts.generate_masks generate_masks.root=court

Notes:
    - Hydra loads configuration from `src/tasks/court_detection/configs/generate_masks.yaml`.
    - The source annotations are expected in `data_{train,val}.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from omegaconf import DictConfig

from src.tasks.base.configuration import require_config_mapping, require_config_value
from src.tasks.court_detection.configuration import validate_paths_boundary
from src.tasks.court_detection.geometry import compute_template_to_image_homography
from src.utils.configuration import PathRole
from src.utils.hydra import hydra_main, register_boundary_validator
from src.utils.schema.court import (
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    HALF_SINGLES_WIDTH,
    SERVICE_LINE_DISTANCE,
)

_xs = HALF_SINGLES_WIDTH
_xd = HALF_DOUBLES_WIDTH
_ys = SERVICE_LINE_DISTANCE
_yB = HALF_LENGTH

# label → (x_min, x_max, y_min, y_max)
CELL_BOUNDS: dict[int, tuple[float, float, float, float]] = {
    1: (-_xs, 0.0, 0.0, _ys),
    2: (0.0, _xs, 0.0, _ys),
    3: (-_xs, 0.0, _ys, _yB),
    4: (0.0, _xs, _ys, _yB),
    5: (-_xd, -_xs, 0.0, _yB),
    6: (_xs, _xd, 0.0, _yB),
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

_BOUNDARY = "court_detection.generate_masks"


def _runtime(cfg: DictConfig) -> tuple[Path, str | None, bool, str]:
    root, resolver = validate_paths_boundary(cfg, expected_sections={"generate_masks"})
    section = require_config_mapping(root, "generate_masks", path="configuration")
    expected = {"root", "image_id", "write_color_mask", "color_dir_name"}
    if set(section) != expected:
        raise ValueError(f"generate_masks requires exactly {sorted(expected)}.")
    root_raw = cast(
        "str", require_config_value(section, "root", str, path="generate_masks")
    )
    image_id = cast(
        "str | None",
        require_config_value(
            section, "image_id", (str, type(None)), path="generate_masks"
        ),
    )
    color_dir_name = cast(
        "str",
        require_config_value(section, "color_dir_name", str, path="generate_masks"),
    )
    if image_id == "":
        raise ValueError("generate_masks.image_id must be null or non-empty.")
    resolver.resolve(PathRole.DATA, root_raw, color_dir_name)
    return (
        resolver.resolve(PathRole.DATA, root_raw),
        image_id,
        cast(
            "bool",
            require_config_value(
                section, "write_color_mask", bool, path="generate_masks"
            ),
        ),
        color_dir_name,
    )


def _validate_boundary(cfg: DictConfig) -> None:
    _runtime(cfg)


register_boundary_validator(_BOUNDARY, _validate_boundary)


def _cell_corners(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> np.ndarray:
    corners: np.ndarray = np.array(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ],
        dtype=np.float32,
    )
    return corners


def _mirror_bounds_origin(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[float, float, float, float]:
    return -x_max, -x_min, -y_max, -y_min


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a label mask (0-6) to a color visualization image (BGR)."""
    color: np.ndarray = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label, bgr in LABEL_COLORS.items():
        color[mask == label] = bgr
    return color


def generate_mask(h: int, w: int, kps_2d: np.ndarray) -> np.ndarray | None:
    """Generate a court cell mask for one image."""
    H = compute_template_to_image_homography(kps_2d, ransac_reproj_threshold=5.0)
    if H is None:
        return None

    mask: np.ndarray = np.zeros((h, w), dtype=np.uint8)

    for label, (x_min, x_max, y_min, y_max) in CELL_BOUNDS.items():
        corners_3d = _cell_corners(x_min, x_max, y_min, y_max)
        corners_img = cast(
            np.ndarray,
            cv2.perspectiveTransform(corners_3d.reshape(1, -1, 2), H)
            .reshape(-1, 2)
            .astype(np.int32),
        )
        cv2.fillPoly(mask, [corners_img], int(label))

        nx_min, nx_max, ny_min, ny_max = _mirror_bounds_origin(
            x_min,
            x_max,
            y_min,
            y_max,
        )
        corners_near = _cell_corners(nx_min, nx_max, ny_min, ny_max)
        corners_near_img = cast(
            np.ndarray,
            cv2.perspectiveTransform(corners_near.reshape(1, -1, 2), H)
            .reshape(-1, 2)
            .astype(np.int32),
        )
        cv2.fillPoly(mask, [corners_near_img], int(label))

    return mask


@hydra_main(
    config_path="../configs",
    config_name="generate_masks",
    version_base="1.3",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Hydra entry point."""
    root, image_id_filter, write_color_mask, color_dir_name = _runtime(cfg)
    masks_dir = root / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    color_masks_dir = root / color_dir_name
    if write_color_mask:
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
            if image_id_filter is not None and entry["id"] != str(image_id_filter):
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
            if write_color_mask:
                color_mask = colorize_mask(mask)
                color_out_path = color_masks_dir / f"{image_id}.png"
                cv2.imwrite(str(color_out_path), color_mask)

            if image_id_filter is not None:
                print(f"[generate_masks] Wrote single sample: {image_id}")
                return 0

    print(
        f"\n[generate_masks] Done: {total} total, "
        f"{total - skipped - errors} written, "
        f"{skipped} skipped, {errors} errors."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
