"""Rendering helpers for clip-level ball detection visualizations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np

from src.tasks.base.visualization.layout import (
    compose_grid,
    label_panel,
    put_text,
)


@dataclass(frozen=True)
class DrawStyle:
    """Marker and text styling for rendered frames."""

    gt_radius: int
    pred_radius: int
    thickness: int
    gt_color_rgb: tuple[int, int, int]
    pred_color_rgb: tuple[int, int, int]
    text_color_rgb: tuple[int, int, int]
    muted_text_color_rgb: tuple[int, int, int]


@dataclass(frozen=True)
class LayoutStyle:
    """Layout settings for rendered visualization frames."""

    header_height: int
    tile_gap: int
    text_scale: float
    text_thickness: int
    background_rgb: tuple[int, int, int]
    panel_label_height: int


def render_animation_frames(
    *,
    frames_rgb: Sequence[np.ndarray],
    frame_names: Sequence[str],
    mdd_frames_rgb: Sequence[np.ndarray],
    pred_coords_px: Sequence[tuple[float, float]],
    pred_visibility: Sequence[bool],
    pred_confidences: Sequence[float],
    pred_heatmaps: Sequence[np.ndarray],
    peak_threshold: float,
    clip_label: str,
    draw: DrawStyle,
    layout: LayoutStyle,
) -> list[np.ndarray]:
    """Render 2x2 animation frames for one clip.

    The four panels are arranged as::

        [ RGB              | MDD                ]
        [ RGB + pred coord | raw heatmap        ]
    """
    rendered_frames: list[np.ndarray] = []
    total_frames = len(frames_rgb)

    for frame_index, (
        frame_rgb,
        frame_name,
        mdd_frame_rgb,
        pred_coord_px,
        pred_is_visible,
        pred_confidence,
        pred_heatmap,
    ) in enumerate(
        zip(
            frames_rgb,
            frame_names,
            mdd_frames_rgb,
            pred_coords_px,
            pred_visibility,
            pred_confidences,
            pred_heatmaps,
            strict=True,
        )
    ):
        top_left = _label(frame_rgb.copy(), text="RGB", draw=draw, layout=layout)
        top_right = _label(
            mdd_frame_rgb.copy(),
            text="MDD (G=brighten, R=darken)",
            draw=draw,
            layout=layout,
        )
        bottom_left = _label(
            _render_prediction_panel(
                frame_rgb=frame_rgb,
                pred_coord_px=pred_coord_px,
                pred_is_visible=pred_is_visible,
                draw=draw,
            ),
            text="RGB + pred coord",
            draw=draw,
            layout=layout,
        )
        bottom_right = _label(
            _render_heatmap_panel(target_hw=frame_rgb.shape[:2], heatmap=pred_heatmap),
            text="raw heatmap",
            draw=draw,
            layout=layout,
        )

        body = compose_grid(
            panels=[[top_left, top_right], [bottom_left, bottom_right]],
            tile_gap=layout.tile_gap,
            background_rgb=layout.background_rgb,
        )
        header = np.full(
            (layout.header_height, body.shape[1], 3),
            layout.background_rgb,
            dtype=np.uint8,
        )

        header_line_1 = (
            f"{clip_label} | frame {frame_index + 1}/{total_frames} | {frame_name}"
        )
        threshold_suffix = "drawn" if pred_is_visible else f"below {peak_threshold:.2f}"
        header_line_2 = f"Pred confidence: {pred_confidence:.3f} ({threshold_suffix})"

        put_text(
            header,
            header_line_1,
            (8, 18),
            color_rgb=draw.text_color_rgb,
            scale=layout.text_scale,
            thickness=layout.text_thickness,
        )
        put_text(
            header,
            header_line_2,
            (8, max(layout.header_height - 10, 30)),
            color_rgb=draw.text_color_rgb
            if pred_is_visible
            else draw.muted_text_color_rgb,
            scale=layout.text_scale,
            thickness=layout.text_thickness,
        )

        rendered_frames.append(np.concatenate([header, body], axis=0))

    return rendered_frames


def _label(
    panel: np.ndarray,
    *,
    text: str,
    draw: DrawStyle,
    layout: LayoutStyle,
) -> np.ndarray:
    return label_panel(
        panel,
        text=text,
        label_height=layout.panel_label_height,
        background_rgb=layout.background_rgb,
        text_color_rgb=draw.text_color_rgb,
        text_scale=layout.text_scale,
        text_thickness=layout.text_thickness,
    )


def _render_prediction_panel(
    *,
    frame_rgb: np.ndarray,
    pred_coord_px: tuple[float, float],
    pred_is_visible: bool,
    draw: DrawStyle,
) -> np.ndarray:
    panel = frame_rgb.copy()
    if pred_is_visible:
        _draw_point(
            panel,
            coord_px=pred_coord_px,
            radius=draw.pred_radius,
            color_rgb=draw.pred_color_rgb,
            thickness=draw.thickness,
        )
    return panel


def _render_heatmap_panel(
    *,
    target_hw: tuple[int, int],
    heatmap: np.ndarray,
) -> np.ndarray:
    """Render the raw (aggregated) heatmap as a standalone colormap panel."""
    heatmap_uint8 = (np.clip(heatmap, 0.0, 1.0) * 255.0).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    heatmap_color = cv2.resize(
        heatmap_color,
        (target_hw[1], target_hw[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return cast("np.ndarray", heatmap_color)


def _draw_point(
    image_rgb: np.ndarray,
    *,
    coord_px: tuple[float, float],
    radius: int,
    color_rgb: tuple[int, int, int],
    thickness: int,
) -> None:
    x_coord = int(round(coord_px[0]))
    y_coord = int(round(coord_px[1]))
    cv2.circle(
        image_rgb,
        (x_coord, y_coord),
        radius,
        color_rgb,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
