"""Inference entry points for ball_detection."""

from src.ball_detection.inference.ensemble_predictor import BallEnsemblePredictor
from src.ball_detection.inference.predictor import BallPredictor
from src.ball_detection.inference.video_api import (
    VideoInferenceConfig,
    VideoInferenceMemberConfig,
    VideoInferenceResult,
    build_video_inference_config,
    run_video_inference,
)

__all__ = [
    "BallPredictor",
    "BallEnsemblePredictor",
    "VideoInferenceMemberConfig",
    "VideoInferenceConfig",
    "VideoInferenceResult",
    "build_video_inference_config",
    "run_video_inference",
]
