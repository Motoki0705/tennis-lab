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


@dataclass(frozen=True)
class PredictionClip:
    """One temporal clip sampled from the full video timeline."""

    frames_rgb: NDArray[np.uint8]
    frame_indices: NDArray[np.int64]


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


def build_prediction_clips(
    inputs: BallDetectionPredictInputs,
    *,
    clip_frames: int | None,
    clip_stride: int | None,
) -> list[PredictionClip]:
    """Split full video inputs into temporal clips."""
    total = int(inputs.frames_rgb.shape[0])
    if total == 0:
        return []

    if clip_frames is None or clip_frames >= total:
        return [
            PredictionClip(
                frames_rgb=inputs.frames_rgb,
                frame_indices=inputs.frame_indices,
            )
        ]

    if clip_frames <= 0:
        raise ValueError(f"clip_frames must be positive or null, got {clip_frames}")

    stride = clip_frames if clip_stride is None else int(clip_stride)
    if stride <= 0:
        raise ValueError(f"clip_stride must be positive or null, got {clip_stride}")

    clips: list[PredictionClip] = []
    for start in range(0, total, stride):
        end = min(start + clip_frames, total)
        if end <= start:
            continue
        clips.append(
            PredictionClip(
                frames_rgb=inputs.frames_rgb[start:end],
                frame_indices=inputs.frame_indices[start:end],
            )
        )

    return clips
