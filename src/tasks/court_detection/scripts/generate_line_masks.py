"""Generate binary white-line masks from court keypoint annotations.

For each annotated image, this script computes a ground-plane homography from
the 14 court keypoints, projects metric-width court line polygons into the
image, and writes a binary mask to ``data/court/line_masks/{id}.png``.

Usage:
    python -m src.tasks.court_detection.scripts.generate_line_masks
    python -m src.tasks.court_detection.scripts.generate_line_masks generate_line_masks.root=data/court

Notes:
    - Hydra loads configuration from `src/tasks/court_detection/configs/generate_line_masks.yaml`.
    - Preview panels are written for sampled images unless `preview_count_per_split=0`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from omegaconf import DictConfig

from src.tasks.court_detection.geometry import compute_template_to_image_homography
from src.utils.hydra import hydra_main
from src.utils.schema.court import (
    CENTER_MARK_LENGTH,
    COURT_SKELETON,
    HALF_LENGTH,
    court_keypoints_3d,
)

LINE_WIDTH_METERS = 0.05
BASELINE_WIDTH_METERS = 0.10
MASK_VALUE = 255


@dataclass(frozen=True)
class MetricLine:
    """A court line segment in court-plane coordinates."""

    start: tuple[float, float]
    end: tuple[float, float]
    width_m: float


def _segment_to_quad(start: np.ndarray, end: np.ndarray, width_m: float) -> np.ndarray:
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        raise ValueError("Degenerate line segment")
    tangent = direction / length
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
    offset = normal * (width_m * 0.5)
    return cast(
        np.ndarray,
        np.stack(
            [start - offset, end - offset, end + offset, start + offset],
            axis=0,
        ).astype(np.float32),
    )


def _center_mark_segments() -> list[MetricLine]:
    return [
        MetricLine((0.0, HALF_LENGTH), (0.0, HALF_LENGTH - CENTER_MARK_LENGTH), LINE_WIDTH_METERS),
        MetricLine((0.0, -HALF_LENGTH), (0.0, -HALF_LENGTH + CENTER_MARK_LENGTH), LINE_WIDTH_METERS),
    ]


def _point_tuple(point: np.ndarray) -> tuple[float, float]:
    return float(point[0]), float(point[1])


def _build_metric_lines() -> list[MetricLine]:
    kp2d = court_keypoints_3d()[:14].numpy()[:, :2]
    baseline_pairs = {(0, 1), (2, 3)}

    lines: list[MetricLine] = []
    for idx_a, idx_b in COURT_SKELETON:
        if idx_a >= 14 or idx_b >= 14:
            continue
        width_m = BASELINE_WIDTH_METERS if (idx_a, idx_b) in baseline_pairs else LINE_WIDTH_METERS
        lines.append(
            MetricLine(
                _point_tuple(kp2d[idx_a]),
                _point_tuple(kp2d[idx_b]),
                width_m,
            )
        )
    lines.extend(_center_mark_segments())
    return lines


METRIC_LINES = _build_metric_lines()


def generate_line_mask(height: int, width: int, kps_2d: np.ndarray) -> np.ndarray | None:
    """Generate a binary white-line mask for one sample."""
    H = compute_template_to_image_homography(kps_2d, ransac_reproj_threshold=5.0)
    if H is None:
        return None

    mask: np.ndarray = np.zeros((height, width), dtype=np.uint8)
    for line in METRIC_LINES:
        quad = _segment_to_quad(
            np.asarray(line.start, dtype=np.float32),
            np.asarray(line.end, dtype=np.float32),
            line.width_m,
        )
        quad_img = cast(
            np.ndarray,
            cv2.perspectiveTransform(quad.reshape(1, -1, 2), H).reshape(-1, 2),
        )
        quad_img = cast(np.ndarray, np.round(quad_img).astype(np.int32))
        cv2.fillPoly(mask, [quad_img], MASK_VALUE)
    return mask


def make_overlay(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend a binary mask on top of the original image."""
    color = np.zeros_like(image_bgr)
    color[..., 2] = mask
    return cast(np.ndarray, cv2.addWeighted(image_bgr, 0.72, color, 0.6, 0.0))


def _sample_preview_ids(entries: list[dict], count: int) -> set[str]:
    if not entries:
        return set()
    take = min(count, len(entries))
    indices = np.linspace(0, len(entries) - 1, take, dtype=int)
    return {str(entries[i]["id"]) for i in indices}


@hydra_main(
    config_path="../configs",
    config_name="generate_line_masks",
    version_base="1.3",
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Hydra entry point."""
    script_cfg = cfg.generate_line_masks
    root = Path(str(script_cfg.root)).expanduser()
    images_dir = root / "images"
    masks_dir = root / str(script_cfg.mask_dir_name)
    masks_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = Path(str(script_cfg.preview_dir)).expanduser()
    preview_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0
    skipped = 0
    errors = 0

    for split in ("train", "val"):
        json_path = root / f"data_{split}.json"
        if not json_path.exists():
            print(f"[generate_line_masks] Missing {json_path}, skipping split.")
            continue

        entries: list[dict] = json.loads(json_path.read_text())
        preview_ids = _sample_preview_ids(
            entries,
            int(script_cfg.preview_count_per_split),
        )
        print(f"[generate_line_masks] Processing {split}: {len(entries)} samples")

        for entry in entries:
            image_id = str(entry["id"])
            image_id_filter = script_cfg.image_id
            if image_id_filter is not None and image_id != str(image_id_filter):
                continue

            total += 1
            img_path = images_dir / f"{image_id}.png"
            if not img_path.exists():
                img_path = images_dir / f"{image_id}.jpg"
            if not img_path.exists():
                print(f"  SKIP (missing image): {image_id}")
                skipped += 1
                continue

            image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                print(f"  ERROR (imread): {image_id}")
                errors += 1
                continue

            kps_2d = np.asarray(entry["kps"], dtype=np.float32)
            mask = generate_line_mask(image_bgr.shape[0], image_bgr.shape[1], kps_2d)
            if mask is None:
                print(f"  ERROR (homography): {image_id}")
                errors += 1
                continue

            cv2.imwrite(str(masks_dir / f"{image_id}.png"), mask)
            written += 1

            if image_id_filter is not None or image_id in preview_ids:
                overlay = make_overlay(image_bgr, mask)
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                panel = np.concatenate([image_bgr, mask_bgr, overlay], axis=1)
                cv2.imwrite(str(preview_dir / f"{split}_{image_id}.png"), panel)

            if image_id_filter is not None:
                print(f"[generate_line_masks] Wrote single sample: {image_id}")
                return 0

    print(
        f"[generate_line_masks] Done: total={total}, written={written}, "
        f"skipped={skipped}, errors={errors}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
