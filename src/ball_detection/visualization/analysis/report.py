"""Reporting helpers for ball_detection visualization workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.ball_detection.inference.video_api import VideoInferenceResult
from src.ball_detection.visualization.types import VideoInputs


def print_video_info(inputs: VideoInputs) -> None:
    """Print loaded video metadata."""
    print("=" * 60)
    print("BALL DETECTION VIDEO INFO")
    print("=" * 60)
    print(f"Frames: {int(inputs.frames_rgb.shape[0])}")
    print(f"Resolution: {int(inputs.width)}x{int(inputs.height)}")
    print(f"FPS: {float(inputs.fps):.3f}")
    print("=" * 60)


def print_prediction_summary(result: VideoInferenceResult) -> None:
    """Print inference summary metrics."""
    num_frames = int(result.frame_indices.shape[0])
    if num_frames == 0:
        print("No predictions generated (0 frames).")
        return

    visible_count = int(np.count_nonzero(result.visibility))
    score = np.nan_to_num(result.score.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
    mean_score = float(np.mean(score)) if score.size > 0 else 0.0
    max_score = float(np.max(score)) if score.size > 0 else 0.0

    print("=" * 60)
    print("BALL DETECTION PREDICTION SUMMARY")
    print("=" * 60)
    print(f"Predicted frames: {num_frames}")
    print(f"Visible frames: {visible_count} ({(visible_count / max(num_frames, 1)) * 100.0:.2f}%)")
    print(f"Score mean: {mean_score:.4f}")
    print(f"Score max: {max_score:.4f}")
    print("=" * 60)


def save_predictions(path: Path, result: VideoInferenceResult) -> Path:
    """Save inference arrays to NPZ artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        frame_indices=result.frame_indices.astype(np.int64, copy=False),
        ball_uv=result.ball_uv.astype(np.float32, copy=False),
        ball_xy_px=result.ball_xy_px.astype(np.float32, copy=False),
        visibility=result.visibility.astype(np.uint8, copy=False),
        score=result.score.astype(np.float32, copy=False),
    )
    return path
