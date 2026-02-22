"""Overlay renderer for ball_detection video predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from src.ball_detection.inference.types import InferenceResult


@dataclass(frozen=True)
class OverlayRenderConfig:
    """Rendering options for prediction overlay video."""

    radius: int
    thickness: int
    color_detected_bgr: tuple[int, int, int]
    color_trail_bgr: tuple[int, int, int]
    show_score: bool
    show_trail: bool
    trail_length: int


def _draw_trail(
    frame_bgr: NDArray[np.uint8],
    trail_points: list[tuple[int, int]],
    *,
    color_bgr: tuple[int, int, int],
) -> None:
    if len(trail_points) < 2:
        return

    total = len(trail_points)
    for idx in range(1, total):
        alpha = float(idx) / float(total)
        color = tuple(int(channel * alpha) for channel in color_bgr)
        thickness = max(1, int(round(1 + 2 * alpha)))
        cv2.line(frame_bgr, trail_points[idx - 1], trail_points[idx], color, thickness, cv2.LINE_AA)


def render_overlay_video(
    *,
    frames_rgb: NDArray[np.uint8],
    predictions: InferenceResult,
    output_path: Path,
    fps: float,
    config: OverlayRenderConfig,
) -> Path:
    """Render ball predictions onto RGB frames and save MP4 overlay video."""
    if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
        raise ValueError(f"frames_rgb must have shape [T, H, W, 3], got {tuple(frames_rgb.shape)}")

    frame_count = min(int(frames_rgb.shape[0]), int(predictions.frame_indices.shape[0]))
    if frame_count == 0:
        raise ValueError("No frames available to render.")

    height = int(frames_rgb.shape[1])
    width = int(frames_rgb.shape[2])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    trail_points: list[tuple[int, int]] = []

    try:
        for frame_idx in range(frame_count):
            frame_bgr = cv2.cvtColor(frames_rgb[frame_idx], cv2.COLOR_RGB2BGR)

            score = float(predictions.score[frame_idx])
            visible = bool(predictions.visibility[frame_idx])
            x = float(predictions.ball_xy_px[frame_idx, 0])
            y = float(predictions.ball_xy_px[frame_idx, 1])

            if visible and np.isfinite(x) and np.isfinite(y):
                xi = int(round(x))
                yi = int(round(y))
                if 0 <= xi < width and 0 <= yi < height:
                    trail_points.append((xi, yi))
                    if len(trail_points) > config.trail_length:
                        trail_points = trail_points[-config.trail_length :]

                    if config.show_trail:
                        _draw_trail(
                            frame_bgr,
                            trail_points,
                            color_bgr=config.color_trail_bgr,
                        )

                    cv2.circle(
                        frame_bgr,
                        (xi, yi),
                        int(config.radius),
                        tuple(int(c) for c in config.color_detected_bgr),
                        int(config.thickness),
                        cv2.LINE_AA,
                    )

            if config.show_score:
                label = f"frame={frame_idx:05d} score={score:.3f}"
                cv2.putText(
                    frame_bgr,
                    label,
                    (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            writer.write(frame_bgr)
    finally:
        writer.release()

    return output_path
