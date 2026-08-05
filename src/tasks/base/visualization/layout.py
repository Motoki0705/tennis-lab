"""Shared panel-composition primitives for raster-family visualizations.

Geometric helpers extracted from the ball-detection clip renderer so that both
ball detection and court detection can lay out labelled multi-panel frames
without duplicating the grid/row/label code.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np


@dataclass(frozen=True)
class PanelStyle:
    """Layout styling shared by composited visualization frames."""

    background_rgb: tuple[int, int, int] = (18, 18, 18)
    text_color_rgb: tuple[int, int, int] = (245, 245, 245)
    text_scale: float = 0.52
    text_thickness: int = 1
    tile_gap: int = 12
    panel_label_height: int = 24


def put_text(
    image_rgb: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    color_rgb: tuple[int, int, int],
    scale: float,
    thickness: int,
) -> None:
    """Draw anti-aliased text onto ``image_rgb`` in place."""
    cv2.putText(
        image_rgb,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color_rgb,
        thickness,
        lineType=cv2.LINE_AA,
    )


def label_panel(
    panel: np.ndarray,
    *,
    text: str,
    label_height: int,
    background_rgb: tuple[int, int, int],
    text_color_rgb: tuple[int, int, int],
    text_scale: float,
    text_thickness: int,
) -> np.ndarray:
    """Prepend a thin label strip above a panel."""
    if label_height <= 0:
        return panel
    label = np.full(
        (label_height, panel.shape[1], 3),
        background_rgb,
        dtype=np.uint8,
    )
    put_text(
        label,
        text,
        (8, max(label_height - 7, 12)),
        color_rgb=text_color_rgb,
        scale=text_scale,
        thickness=text_thickness,
    )
    return cast("np.ndarray", np.concatenate([label, panel], axis=0))


def compose_row(
    *,
    panels: Sequence[np.ndarray],
    tile_gap: int,
    background_rgb: tuple[int, int, int],
) -> np.ndarray:
    """Lay panels of equal size out horizontally with a uniform gap."""
    if not panels:
        raise ValueError("At least one panel is required.")
    panel_height = panels[0].shape[0]
    panel_width = panels[0].shape[1]
    canvas_width = len(panels) * panel_width + (len(panels) - 1) * tile_gap
    canvas = np.full((panel_height, canvas_width, 3), background_rgb, dtype=np.uint8)

    cursor_x = 0
    for panel in panels:
        canvas[:, cursor_x : cursor_x + panel_width] = panel
        cursor_x += panel_width + tile_gap
    return cast("np.ndarray", canvas)


def compose_grid(
    *,
    panels: Sequence[Sequence[np.ndarray]],
    tile_gap: int,
    background_rgb: tuple[int, int, int],
) -> np.ndarray:
    """Lay rows of equal-width panels out vertically with a uniform gap."""
    if not panels or not panels[0]:
        raise ValueError("At least one panel is required.")

    row_images = [
        compose_row(panels=row, tile_gap=tile_gap, background_rgb=background_rgb)
        for row in panels
    ]

    row_width = row_images[0].shape[1]
    canvas_height = (
        sum(row.shape[0] for row in row_images) + (len(row_images) - 1) * tile_gap
    )
    canvas = np.full((canvas_height, row_width, 3), background_rgb, dtype=np.uint8)

    cursor_y = 0
    for row in row_images:
        canvas[cursor_y : cursor_y + row.shape[0], :] = row
        cursor_y += row.shape[0] + tile_gap
    return cast("np.ndarray", canvas)
