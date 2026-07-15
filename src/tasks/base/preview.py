"""Shared helpers for Hydra-driven dataset preview scripts.

These parse the common ``cfg.preview`` / ``cfg.data.split`` config conventions
used by the per-task ``preview_heatmaps`` / ``preview_augmentation`` scripts.
They depend only on the OmegaConf config shape, not on any task's domain types,
so the ball- and court-detection scripts share a single implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import cv2
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import Tensor

from src.tasks.base.data.court_lines import (
    CourtLineFrameResult,
    CourtLineInputBuilder,
    CourtLineInputConfig,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = [
    "build_court_line_preview_rows",
    "compose_titled_row",
    "court_line_frame_metadata",
    "draw_normalized_point",
    "enable_all_augmentation_blocks",
    "make_court_line_preview_builder",
    "make_court_kp_preview_config",
    "render_court_line_frame",
    "resolve_court_input_type",
    "resolve_sample_indices",
    "resolve_split_file",
]


def resolve_court_input_type(cfg: DictConfig) -> str:
    """Return the validated preview court modality (``kp`` or ``line``)."""
    court_input_type = str(cfg.preview.court_input_type)
    if court_input_type not in {"kp", "line"}:
        raise ValueError(
            "preview.court_input_type must be 'kp' or 'line', got "
            f"{court_input_type!r}."
        )
    return court_input_type


def make_court_kp_preview_config(cfg: DictConfig) -> DictConfig:
    """Clone a preview config while forcing datasets to expose source court KP.

    Line previews need the projected CourtKP20 source in order to render and
    corrupt the same line map used during training. The returned clone leaves
    the caller's config untouched and changes only ``data.court_input_type``.
    """
    cloned = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    if not isinstance(cloned, DictConfig) or not isinstance(cloned.data, DictConfig):
        raise ValueError("Preview config must contain a data mapping.")
    with open_dict(cloned.data):
        cloned.data.court_input_type = "kp"
    return cloned


def make_court_line_preview_builder(cfg: DictConfig) -> CourtLineInputBuilder:
    """Build the line-map pipeline from optional preview overrides."""
    raw = OmegaConf.to_container(cfg.preview.court_line, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("preview.court_line must be a mapping.")
    return CourtLineInputBuilder(CourtLineInputConfig.from_mapping(raw))


def build_court_line_preview_rows(
    builder: CourtLineInputBuilder,
    court_kp: Tensor,
    *,
    original_seed: int,
    variant_seeds: Sequence[int],
) -> list[list[CourtLineFrameResult]]:
    """Build clean and augmented line observations for every camera."""
    if court_kp.ndim != 4 or tuple(court_kp.shape[-2:]) != (20, 2):
        raise ValueError(
            f"court_kp must have shape (V,T,20,2), got {tuple(court_kp.shape)}."
        )
    if int(court_kp.shape[1]) < 1:
        raise ValueError("court_kp must contain at least one frame.")

    def build_row(*, augment: bool, seed: int) -> list[CourtLineFrameResult]:
        rng = np.random.default_rng(seed)
        return [
            builder.build_frame(court_kp[camera_index, 0], augment=augment, rng=rng)
            for camera_index in range(int(court_kp.shape[0]))
        ]

    rows = [build_row(augment=False, seed=original_seed)]
    rows.extend(build_row(augment=True, seed=int(seed)) for seed in variant_seeds)
    return rows


def court_line_frame_metadata(frame: CourtLineFrameResult) -> dict[str, int | float]:
    """Serialize extractor diagnostics without adding them to model inputs."""
    diagnostics = frame.extraction.diagnostics
    return {
        "input_point_count": diagnostics.input_point_count,
        "retained_point_count": diagnostics.retained_point_count,
        "extracted_line_count": diagnostics.extracted_line_count,
        "mean_inlier_ratio": diagnostics.mean_inlier_ratio,
        "mean_residual_px": diagnostics.mean_residual_px,
        "line_coverage": diagnostics.line_coverage,
    }


def render_court_line_frame(ax: Axes, frame: CourtLineFrameResult) -> None:
    """Render a degraded line map and the normalized RANSAC finite segments."""
    ax.imshow(
        frame.line_map,
        cmap="gray",
        vmin=0,
        vmax=255,
        interpolation="nearest",
        extent=(0.0, 1.0, 1.0, 0.0),
    )
    segments = frame.extraction.segments
    valid = segments[np.any(segments != 0.0, axis=1)]
    for u1, v1, u2, v2 in valid:
        ax.plot((u1, u2), (v1, v2), color="#00e5ff", linewidth=1.8)
        ax.scatter((u1, u2), (v1, v2), color="#ffca28", s=8, zorder=4)
    ax.text(
        0.02,
        0.04,
        f"RANSAC lines: {len(valid)}",
        color="white",
        fontsize=7,
        transform=ax.transAxes,
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
    )


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
