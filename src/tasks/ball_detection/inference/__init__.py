"""Inference entry points for ball_detection."""

from src.tasks.ball_detection.inference.adapters import (
    HRNetContextInputAdapter,
    ModelInputAdapter,
    TrackNetV3InputAdapter,
    build_adapter_for_model,
)
from src.tasks.ball_detection.inference.config import build_inference_config
from src.tasks.ball_detection.inference.ensemble_predictor import BallEnsemblePredictor
from src.tasks.ball_detection.inference.predictor import BallPredictor
from src.tasks.ball_detection.inference.types import (
    InferenceConfig,
    InferenceMemberConfig,
    InferenceResult,
)

__all__ = [
    "BallPredictor",
    "BallEnsemblePredictor",
    "ModelInputAdapter",
    "TrackNetV3InputAdapter",
    "HRNetContextInputAdapter",
    "build_adapter_for_model",
    "InferenceMemberConfig",
    "InferenceConfig",
    "InferenceResult",
    "build_inference_config",
]
