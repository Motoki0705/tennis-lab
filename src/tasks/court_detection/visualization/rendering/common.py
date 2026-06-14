"""Shared rendering primitives for court-detection visualization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import cv2
import numpy as np

from src.tasks.base.visualization.layout import (
    PanelStyle,
    compose_row,
    label_panel,
    put_text,
)

# RGB palette for the 7 court segmentation classes (class 0 = background).
COURT_SEG_PALETTE_RGB: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (230, 75, 60),
    (60, 120, 230),
    (60, 200, 90),
    (240, 200, 50),
    (200, 70, 220),
    (70, 210, 220),
)


@dataclass(frozen=True)
class CourtRenderStyle:
    """Styling for court visualization panels."""

    panel: PanelStyle = field(default_factory=PanelStyle)
    header_height: int = 36
    display_width: int = 640
    kp_radius: int = 4
    kp_color_rgb: tuple[int, int, int] = (96, 255, 128)
    kp_thickness: int = -1
    line_threshold: float = 0.5


def resize_for_display(rgb: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale an RGB frame so its width is at most ``max_width``."""
    height, width = rgb.shape[:2]
    if width <= max_width:
        return rgb
    scale = max_width / float(width)
    new_size = (max_width, max(int(round(height * scale)), 1))
    return cast("np.ndarray", cv2.resize(rgb, new_size, interpolation=cv2.INTER_LINEAR))


def colorize_seg_mask(mask: np.ndarray) -> np.ndarray:
    """Map an integer class mask ``(H, W)`` to an RGB image."""
    height, width = mask.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for label, color in enumerate(COURT_SEG_PALETTE_RGB):
        rgb[mask == label] = color
    return cast("np.ndarray", rgb)


def colorize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Map a ``[0, 1]`` heatmap to a JET-colormap RGB image."""
    heatmap_uint8 = (np.clip(heatmap, 0.0, 1.0) * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cast("np.ndarray", cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))


def line_prob_to_map(prob: np.ndarray) -> np.ndarray:
    """Render a line-probability map ``(H, W)`` as white-on-black RGB."""
    intensity = (np.clip(prob, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cast("np.ndarray", cv2.cvtColor(intensity, cv2.COLOR_GRAY2RGB))


def compose_two_panel(
    *,
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    left_label: str,
    right_label: str,
    header_text: str,
    style: CourtRenderStyle,
) -> np.ndarray:
    """Compose a labelled ``[left | right]`` panel with a header strip.

    ``right_rgb`` is resized to match ``left_rgb`` before composing.
    """
    height, width = left_rgb.shape[:2]
    if right_rgb.shape[:2] != (height, width):
        right_rgb = cv2.resize(right_rgb, (width, height), interpolation=cv2.INTER_NEAREST)

    panel = style.panel
    left = label_panel(
        left_rgb,
        text=left_label,
        label_height=panel.panel_label_height,
        background_rgb=panel.background_rgb,
        text_color_rgb=panel.text_color_rgb,
        text_scale=panel.text_scale,
        text_thickness=panel.text_thickness,
    )
    right = label_panel(
        right_rgb,
        text=right_label,
        label_height=panel.panel_label_height,
        background_rgb=panel.background_rgb,
        text_color_rgb=panel.text_color_rgb,
        text_scale=panel.text_scale,
        text_thickness=panel.text_thickness,
    )
    body = compose_row(
        panels=[left, right],
        tile_gap=panel.tile_gap,
        background_rgb=panel.background_rgb,
    )
    header = np.full(
        (style.header_height, body.shape[1], 3),
        panel.background_rgb,
        dtype=np.uint8,
    )
    put_text(
        header,
        header_text,
        (8, max(style.header_height - 12, 16)),
        color_rgb=panel.text_color_rgb,
        scale=panel.text_scale,
        thickness=panel.text_thickness,
    )
    return cast("np.ndarray", np.concatenate([header, body], axis=0))
