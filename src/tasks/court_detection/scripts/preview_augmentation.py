"""Render original-vs-augmented previews for court-detection training samples.

Usage:
    python -m src.tasks.court_detection.scripts.preview_augmentation
    python -m src.tasks.court_detection.scripts.preview_augmentation data=court_seg
    python -m src.tasks.court_detection.scripts.preview_augmentation data=court_line preview.split=val
    python -m src.tasks.court_detection.scripts.preview_augmentation preview.sample_indices=[0,8,16]

Notes:
    - Hydra loads configuration from `src/tasks/court_detection/configs/preview_augmentation.yaml`.
    - Court augmentation has no per-transform `enabled` flags; the "augmented"
      panels are produced by building the dataset with `is_train=True` (full
      training pipeline) while the "original" panel uses `is_train=False`
      (deterministic validation resize only).
    - Each sample renders one original panel plus `preview.num_augmented`
      independently seeded augmentation draws, with task-specific annotations
      (kp: keypoint circles, seg/line: mask overlay).
    - Outputs are written under `outputs/court_detection/augmentation_preview`.
"""

from __future__ import annotations

import random
import sys
from collections.abc import Callable, Sized
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from omegaconf import DictConfig

from src.tasks.base.preview import compose_titled_row, resolve_sample_indices
from src.tasks.court_detection.data.datamodule import CourtDetectionDataModule
from src.tasks.court_detection.visualization.rendering.common import (
    colorize_seg_mask,
    denormalize_tensor_to_rgb,
)
from src.utils.hydra import hydra_main
from src.utils.io import save_json

_LINE_OVERLAY_RGB = (255, 96, 96)
_KP_COLOR_RGB = (255, 80, 80)


@hydra_main(
    config_path="../configs",
    config_name="preview_augmentation",
    version_base="1.3",
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Hydra entry point."""
    output_dir = Path(str(cfg.preview.output_dir)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_name = str(cfg.preview.split)
    datamodule = CourtDetectionDataModule(cfg)
    base_dataset = datamodule.create_dataset(split=split_name, is_train=False)
    augmented_dataset = datamodule.create_dataset(split=split_name, is_train=True)

    task = str(cfg.data.task)
    seed = int(cfg.preview.seed)
    num_augmented = int(cfg.preview.num_augmented)
    if num_augmented < 1:
        raise ValueError("preview.num_augmented must be >= 1.")

    sample_indices = resolve_sample_indices(
        cfg, dataset_size=len(cast("Sized", base_dataset)), min_samples=1
    )
    manifest: list[dict[str, Any]] = []
    for sample_index in sample_indices:
        _seed_all(seed + sample_index)
        base_sample = base_dataset[sample_index]
        panels = [_annotate_sample(base_sample, task=task, cfg=cfg)]
        titles = ["original"]
        for variant in range(num_augmented):
            _seed_all(seed + sample_index * 1009 + variant + 1)
            augmented_sample = augmented_dataset[sample_index]
            panels.append(_annotate_sample(augmented_sample, task=task, cfg=cfg))
            titles.append(f"augmented #{variant}")

        panels = _pad_panels_to_common_size(panels, cfg)
        sheet = compose_titled_row(panels, titles, cfg)

        image_id = str(base_sample["image_id"])
        file_stem = f"{sample_index:06d}_{image_id}"
        image_path = output_dir / f"{file_stem}.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

        metadata = {
            "sample_index": sample_index,
            "image_id": image_id,
            "task": task,
            "split": split_name,
            "num_augmented": num_augmented,
            "output_image": str(image_path),
        }
        save_json(metadata, output_dir / f"{file_stem}.json")
        manifest.append(metadata)

    save_json(manifest, output_dir / "manifest.json")
    print(f"Saved {len(manifest)} augmentation preview(s) to {output_dir}")
    return 0


def _seed_all(seed: int) -> None:
    """Seed every RNG used by the court augmentation pipeline."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def _annotate_sample(
    sample: dict[str, Any],
    *,
    task: str,
    cfg: DictConfig,
) -> np.ndarray:
    """Render one dataset sample as an RGB panel with its annotations."""
    image = cast("torch.Tensor", sample["image"])
    rgb = denormalize_tensor_to_rgb(image)
    if task == "kp":
        return _overlay_keypoints(rgb, sample, cfg)
    if task == "seg":
        return _overlay_seg_mask(rgb, sample, cfg)
    if task == "line":
        return _overlay_line_mask(rgb, sample, cfg)
    raise ValueError(f"Unknown data.task: {task!r}")


def _overlay_keypoints(
    rgb: np.ndarray, sample: dict[str, Any], cfg: DictConfig
) -> np.ndarray:
    """Draw pixel-space court keypoints onto the panel."""
    overlay = rgb.copy()
    keypoints = cast("torch.Tensor", sample["keypoints"]).cpu().numpy()
    height, width = overlay.shape[:2]
    for x_pos, y_pos in keypoints:
        if not (0.0 <= float(x_pos) < width and 0.0 <= float(y_pos) < height):
            continue
        cv2.circle(
            overlay,
            (int(round(float(x_pos))), int(round(float(y_pos)))),
            int(cfg.preview.draw.kp_radius),
            _KP_COLOR_RGB,
            thickness=int(cfg.preview.draw.kp_thickness),
            lineType=cv2.LINE_AA,
        )
    return overlay


def _overlay_seg_mask(
    rgb: np.ndarray, sample: dict[str, Any], cfg: DictConfig
) -> np.ndarray:
    """Blend the colorized segmentation mask over foreground pixels."""
    mask = cast("torch.Tensor", sample["mask"]).cpu().numpy()
    colored = colorize_seg_mask(mask)
    return _blend_where(rgb, colored, mask > 0, float(cfg.preview.draw.mask_alpha))


def _overlay_line_mask(
    rgb: np.ndarray, sample: dict[str, Any], cfg: DictConfig
) -> np.ndarray:
    """Tint white-line pixels with a solid overlay color."""
    mask = cast("torch.Tensor", sample["mask"]).cpu().numpy()[0]
    colored = np.zeros_like(rgb)
    colored[:, :] = _LINE_OVERLAY_RGB
    return _blend_where(rgb, colored, mask > 0.5, float(cfg.preview.draw.mask_alpha))


def _blend_where(
    rgb: np.ndarray, colored: np.ndarray, region: np.ndarray, alpha: float
) -> np.ndarray:
    """Alpha-blend ``colored`` into ``rgb`` only where ``region`` is true."""
    overlay = rgb.copy()
    blended = rgb.astype(np.float32) * (1.0 - alpha) + colored.astype(np.float32) * alpha
    overlay[region] = np.clip(blended[region], 0.0, 255.0).astype(np.uint8)
    return overlay


def _pad_panels_to_common_size(
    panels: list[np.ndarray], cfg: DictConfig
) -> list[np.ndarray]:
    """Pad panels to a shared canvas so they can be composed in one row.

    Augmented crops/resizes produce per-panel sizes, while the shared row
    composer requires equal panels; padding (instead of resizing) keeps the
    augmented geometry visible at true scale.
    """
    background_rgb = tuple(int(v) for v in cfg.preview.layout.background_rgb)
    max_height = max(panel.shape[0] for panel in panels)
    max_width = max(panel.shape[1] for panel in panels)
    padded: list[np.ndarray] = []
    for panel in panels:
        canvas = np.full((max_height, max_width, 3), background_rgb, dtype=np.uint8)
        canvas[: panel.shape[0], : panel.shape[1]] = panel
        padded.append(canvas)
    return padded


if __name__ == "__main__":
    sys.exit(cast(Callable[[], int], main)())
