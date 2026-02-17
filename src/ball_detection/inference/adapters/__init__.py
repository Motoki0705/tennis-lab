"""Temporal input adapters for ball_detection inference."""

from src.ball_detection.inference.adapters.base import ModelInputAdapter
from src.ball_detection.inference.adapters.factory import build_adapter_for_model
from src.ball_detection.inference.adapters.hrnet import HRNetContextInputAdapter
from src.ball_detection.inference.adapters.tracknetv3 import TrackNetV3InputAdapter

__all__ = [
    "ModelInputAdapter",
    "TrackNetV3InputAdapter",
    "HRNetContextInputAdapter",
    "build_adapter_for_model",
]
