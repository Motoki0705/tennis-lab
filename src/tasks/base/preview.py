"""Shared helpers for Hydra-driven dataset preview scripts.

These parse the common ``cfg.preview`` / ``cfg.data.split`` config conventions
used by the per-task ``preview_heatmaps`` / ``preview_augmentation`` scripts.
They depend only on the OmegaConf config shape, not on any task's domain types,
so the ball- and court-detection scripts share a single implementation.
"""

from __future__ import annotations

from typing import Any, cast

import cv2
import numpy as np
from omegaconf import DictConfig, OmegaConf

__all__ = [
    "compose_titled_row",
    "draw_normalized_point",
    "enable_all_augmentation_blocks",
    "resolve_sample_indices",
    "resolve_split_file",
]


def enable_all_augmentation_blocks(augmentation_cfg: DictConfig) -> dict[str, Any]:
    """Return a plain-dict copy of ``augmentation_cfg`` with every block enabled.

    Sets the top-level ``enabled`` flag and each per-block ``enabled`` flag to
    ``True`` while leaving all other parameters (probabilities, magnitudes)
    untouched, so preview scripts exercise every configured transform.
    """
    container = OmegaConf.to_container(augmentation_cfg, resolve=True)
    if not isinstance(container, dict):
        raise ValueError(
            f"augmentation config must be a mapping, got {type(container).__name__}."
        )
    container["enabled"] = True
    for block in container.values():
        if isinstance(block, dict) and "enabled" in block:
            block["enabled"] = True
    return cast("dict[str, Any]", container)


def resolve_split_file(cfg: DictConfig, split_name: str) -> str:
    """Return the split-file path for ``split_name`` from ``cfg.data.split``."""
    split_cfg = cfg.data.split
    key = f"{split_name}_file"
    if key not in split_cfg:
        available = ", ".join(sorted(split_cfg.keys()))
        raise ValueError(f"Unknown preview.split={split_name!r}. Available: {available}")
    return str(split_cfg[key])


def resolve_sample_indices(
    cfg: DictConfig,
    dataset_size: int,
    *,
    min_samples: int = 0,
) -> list[int]:
    """Return validated preview sample indices.

    Uses ``cfg.preview.sample_indices`` when non-empty, otherwise the first
    ``max(cfg.preview.max_samples, min_samples)`` indices (clamped to
    ``dataset_size``). ``min_samples`` defaults to ``0`` (no floor); pass ``1``
    to guarantee at least one sample. Raises ``IndexError`` if any resolved
    index is out of range.
    """
    explicit = [int(value) for value in cfg.preview.sample_indices]
    if explicit:
        sample_indices = explicit
    else:
        count = max(int(cfg.preview.max_samples), min_samples)
        sample_indices = list(range(min(count, dataset_size)))
    for sample_index in sample_indices:
        if sample_index < 0 or sample_index >= dataset_size:
            raise IndexError(
                f"preview sample_index={sample_index} is out of range for "
                f"dataset size {dataset_size}."
            )
    return sample_indices


def draw_normalized_point(
    image: np.ndarray,
    center_xy: tuple[float, float],
    *,
    radius: int,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw a circle at normalized ``(x, y)`` coordinates on ``image`` in place."""
    height, width = image.shape[:2]
    x_px = int(round(center_xy[0] * max(width - 1, 0)))
    y_px = int(round(center_xy[1] * max(height - 1, 0)))
    cv2.circle(image, (x_px, y_px), radius, color, thickness=thickness, lineType=cv2.LINE_AA)


def compose_titled_row(
    panels: list[np.ndarray],
    titles: list[str],
    cfg: DictConfig,
) -> np.ndarray:
    """Compose panels side by side with per-panel titles in a header strip."""
    tile_gap = int(cfg.preview.layout.tile_gap)
    header_height = int(cfg.preview.layout.header_height)
    text_scale = float(cfg.preview.layout.text_scale)
    text_thickness = int(cfg.preview.layout.text_thickness)
    background_rgb = tuple(int(v) for v in cfg.preview.layout.background_rgb)

    height, width = panels[0].shape[:2]
    row_width = len(panels) * width + (len(panels) - 1) * tile_gap
    canvas = np.full((header_height + height, row_width, 3), background_rgb, dtype=np.uint8)

    cursor_x = 0
    for panel, title in zip(panels, titles, strict=True):
        canvas[header_height:, cursor_x : cursor_x + width] = panel
        cv2.putText(
            canvas,
            title,
            (cursor_x + 6, max(header_height - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (245, 245, 245),
            text_thickness,
            lineType=cv2.LINE_AA,
        )
        cursor_x += width + tile_gap
    return cast("np.ndarray", canvas)
