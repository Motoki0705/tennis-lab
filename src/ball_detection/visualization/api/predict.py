"""Prediction API for ball_detection visualization."""

from __future__ import annotations

from src.ball_detection.inference.video_api import (
    VideoInferenceConfig,
    VideoInferenceResult,
    run_video_inference,
)
from src.ball_detection.visualization.adapters.predict_inputs import BallDetectionPredictInputs


def predict_video(
    *,
    inputs: BallDetectionPredictInputs,
    inference_config: VideoInferenceConfig,
) -> VideoInferenceResult:
    """Run configured video inference and return per-frame predictions."""
    return run_video_inference(frames_rgb=inputs.frames_rgb, config=inference_config)
