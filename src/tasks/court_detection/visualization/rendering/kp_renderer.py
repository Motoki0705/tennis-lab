"""Keypoint visualization renderer for court detection."""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np

from src.tasks.court_detection.visualization.api.predict import KpFramePrediction
from src.tasks.court_detection.visualization.io.frames import CourtFrame
from src.tasks.court_detection.visualization.rendering.common import (
    CourtRenderStyle,
    colorize_heatmap,
    compose_two_panel,
    resize_for_display,
)


def render_kp_frames(
    *,
    frames: list[CourtFrame],
    predictions: list[KpFramePrediction],
    style: CourtRenderStyle,
    clip_label: str,
) -> list[np.ndarray]:
    """Render ``[RGB + predicted keypoints | mean heatmap]`` frames."""
    total = len(frames)
    rendered: list[np.ndarray] = []
    for index, (frame, prediction) in enumerate(zip(frames, predictions, strict=True)):
        left = resize_for_display(frame.rgb, style.display_width).copy()
        _draw_keypoints(
            left,
            keypoints_px=prediction.keypoints_px,
            original_hw=frame.rgb.shape[:2],
            style=style,
        )
        right = colorize_heatmap(_normalize_for_display(prediction.mean_heatmap))
        header = f"{clip_label} | frame {index + 1}/{total} | {frame.name}"
        rendered.append(
            compose_two_panel(
                left_rgb=left,
                right_rgb=right,
                left_label="RGB + pred KP",
                right_label="mean heatmap",
                header_text=header,
                style=style,
            )
        )
    return rendered


def _normalize_for_display(heatmap: np.ndarray) -> np.ndarray:
    """Scale an averaged heatmap to ``[0, 1]`` so faint peaks stay visible."""
    peak = float(heatmap.max())
    if peak <= 0.0:
        return heatmap
    return cast("np.ndarray", heatmap / peak)


def _draw_keypoints(
    image_rgb: np.ndarray,
    *,
    keypoints_px: np.ndarray,
    original_hw: tuple[int, int],
    style: CourtRenderStyle,
) -> None:
    original_height, original_width = original_hw
    display_height, display_width = image_rgb.shape[:2]
    scale_x = display_width / max(original_width, 1)
    scale_y = display_height / max(original_height, 1)
    for x_coord, y_coord in keypoints_px:
        cv2.circle(
            image_rgb,
            (int(round(float(x_coord) * scale_x)), int(round(float(y_coord) * scale_y))),
            style.kp_radius,
            style.kp_color_rgb,
            thickness=style.kp_thickness,
            lineType=cv2.LINE_AA,
        )
