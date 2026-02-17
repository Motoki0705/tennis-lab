"""Adapters for building predictor inputs from loaded video frames."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.ball_detection.visualization.types import VideoInputs


@dataclass(frozen=True)
class BallDetectionPredictInputs:
    """Predictor-ready payload for one video."""

    frames_rgb: NDArray[np.uint8]
    frame_indices: NDArray[np.int64]
    width: int
    height: int
    fps: float


def build_ball_detection_predict_inputs(inputs: VideoInputs) -> BallDetectionPredictInputs:
    """Normalize loaded video inputs for predictor API."""
    frames_rgb = np.asarray(inputs.frames_rgb)
    if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
        raise ValueError(f"Expected video frames shape [T, H, W, 3], got {tuple(frames_rgb.shape)}")

    if frames_rgb.dtype != np.uint8:
        frames_rgb = np.clip(frames_rgb, 0, 255).astype(np.uint8)

    frames_rgb = np.ascontiguousarray(frames_rgb)
    frame_indices = np.arange(frames_rgb.shape[0], dtype=np.int64)

    return BallDetectionPredictInputs(
        frames_rgb=frames_rgb,
        frame_indices=frame_indices,
        width=int(inputs.width),
        height=int(inputs.height),
        fps=float(inputs.fps),
    )
