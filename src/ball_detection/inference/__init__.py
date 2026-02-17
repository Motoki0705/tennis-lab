"""Inference entry points for ball_detection."""

from src.ball_detection.inference.ensemble_predictor import (
    BallEnsemblePredictor,
    HRNetContextInputAdapter,
    ModelInputAdapter,
    TrackNetV3InputAdapter,
)
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
    "ModelInputAdapter",
    "TrackNetV3InputAdapter",
    "HRNetContextInputAdapter",
    "VideoInferenceMemberConfig",
    "VideoInferenceConfig",
    "VideoInferenceResult",
    "build_video_inference_config",
    "run_video_inference",
]
