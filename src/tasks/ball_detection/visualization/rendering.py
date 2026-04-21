"""Rendering helpers for clip-level ball detection visualizations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from PIL import Image


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
    heatmap_alpha: float
    show_heatmap_panel: bool


def render_animation_frames(
    *,
    frames_rgb: Sequence[np.ndarray],
    frame_names: Sequence[str],
    gt_coords_px: Sequence[tuple[float, float]],
    gt_visibility: Sequence[bool],
    pred_coords_px: Sequence[tuple[float, float]],
    pred_visibility: Sequence[bool],
    pred_confidences: Sequence[float],
    pred_heatmaps: Sequence[np.ndarray],
    peak_threshold: float,
    clip_label: str,
    draw: DrawStyle,
    layout: LayoutStyle,
) -> list[np.ndarray]:
    """Render animation frames for one clip."""
    rendered_frames: list[np.ndarray] = []
    total_frames = len(frames_rgb)

    for frame_index, (
        frame_rgb,
        frame_name,
        gt_coord_px,
        gt_is_visible,
        pred_coord_px,
        pred_is_visible,
        pred_confidence,
        pred_heatmap,
    ) in enumerate(
        zip(
            frames_rgb,
            frame_names,
            gt_coords_px,
            gt_visibility,
            pred_coords_px,
            pred_visibility,
            pred_confidences,
            pred_heatmaps,
            strict=True,
        )
    ):
        panels = [
            _render_prediction_panel(
                frame_rgb=frame_rgb,
                gt_coord_px=gt_coord_px,
                gt_is_visible=gt_is_visible,
                pred_coord_px=pred_coord_px,
                pred_is_visible=pred_is_visible,
                draw=draw,
            )
        ]

        if layout.show_heatmap_panel:
            panels.append(
                _render_heatmap_panel(
                    frame_rgb=frame_rgb,
                    heatmap=pred_heatmap,
                    gt_coord_px=gt_coord_px,
                    gt_is_visible=gt_is_visible,
                    pred_coord_px=pred_coord_px,
                    pred_is_visible=pred_is_visible,
                    draw=draw,
                    layout=layout,
                )
            )

        body = _compose_panels(
            panels=panels,
            tile_gap=layout.tile_gap,
            background_rgb=layout.background_rgb,
        )
        header = np.full(
            (layout.header_height, body.shape[1], 3),
            layout.background_rgb,
            dtype=np.uint8,
        )

        header_line_1 = f"{clip_label} | frame {frame_index + 1}/{total_frames} | {frame_name}"
        visibility_text = "visible" if gt_is_visible else "hidden"
        threshold_suffix = "drawn" if pred_is_visible else f"below {peak_threshold:.2f}"
        header_line_2 = (
            f"GT: {visibility_text} | Pred confidence: {pred_confidence:.3f} ({threshold_suffix})"
        )

        cv2.putText(
            header,
            header_line_1,
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            layout.text_scale,
            draw.text_color_rgb,
            layout.text_thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            header,
            header_line_2,
            (8, max(layout.header_height - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            layout.text_scale,
            draw.text_color_rgb if pred_is_visible else draw.muted_text_color_rgb,
            layout.text_thickness,
            lineType=cv2.LINE_AA,
        )

        rendered_frames.append(np.concatenate([header, body], axis=0))

    return rendered_frames


def save_gif(
    *,
    frames_rgb: Sequence[np.ndarray],
    path: Path,
    fps: float,
    loop: int = 0,
) -> None:
    """Save rendered RGB frames as an animated GIF."""
    if not frames_rgb:
        raise ValueError("At least one frame is required to save a GIF.")
    if fps <= 0:
        raise ValueError("fps must be positive.")
    if path.suffix.lower() != ".gif":
        raise ValueError(f"Only .gif outputs are supported, got: {path}")

    duration_ms = max(int(round(1000.0 / fps)), 1)
    pil_frames = [
        Image.fromarray(frame).convert(
            "P",
            palette=Image.Palette.ADAPTIVE,
            colors=256,
        )
        for frame in frames_rgb
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=loop,
        disposal=2,
        optimize=True,
    )


def _render_prediction_panel(
    *,
    frame_rgb: np.ndarray,
    gt_coord_px: tuple[float, float],
    gt_is_visible: bool,
    pred_coord_px: tuple[float, float],
    pred_is_visible: bool,
    draw: DrawStyle,
) -> np.ndarray:
    panel = frame_rgb.copy()
    if gt_is_visible:
        _draw_point(
            panel,
            coord_px=gt_coord_px,
            radius=draw.gt_radius,
            color_rgb=draw.gt_color_rgb,
            thickness=draw.thickness,
        )
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
    frame_rgb: np.ndarray,
    heatmap: np.ndarray,
    gt_coord_px: tuple[float, float],
    gt_is_visible: bool,
    pred_coord_px: tuple[float, float],
    pred_is_visible: bool,
    draw: DrawStyle,
    layout: LayoutStyle,
) -> np.ndarray:
    panel = frame_rgb.copy()
    heatmap_uint8 = np.clip(heatmap, 0.0, 1.0)
    heatmap_uint8 = (heatmap_uint8 * 255.0).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    heatmap_color = cv2.resize(
        heatmap_color,
        (panel.shape[1], panel.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    panel = cv2.addWeighted(panel, 1.0 - layout.heatmap_alpha, heatmap_color, layout.heatmap_alpha, 0.0)

    if gt_is_visible:
        _draw_point(
            panel,
            coord_px=gt_coord_px,
            radius=draw.gt_radius,
            color_rgb=draw.gt_color_rgb,
            thickness=draw.thickness,
        )
    if pred_is_visible:
        _draw_point(
            panel,
            coord_px=pred_coord_px,
            radius=draw.pred_radius,
            color_rgb=draw.pred_color_rgb,
            thickness=draw.thickness,
        )

    return cast(np.ndarray, panel)


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


def _compose_panels(
    *,
    panels: Sequence[np.ndarray],
    tile_gap: int,
    background_rgb: tuple[int, int, int],
) -> np.ndarray:
    if not panels:
        raise ValueError("At least one panel is required.")

    panel_height = panels[0].shape[0]
    panel_width = panels[0].shape[1]
    canvas_width = len(panels) * panel_width + (len(panels) - 1) * tile_gap
    canvas = np.full(
        (panel_height, canvas_width, 3),
        background_rgb,
        dtype=np.uint8,
    )

    cursor_x = 0
    for panel in panels:
        canvas[:, cursor_x : cursor_x + panel_width] = panel
        cursor_x += panel_width + tile_gap
    return cast(np.ndarray, canvas)
