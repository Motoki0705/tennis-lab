"""Inference predictors for event detection."""

from src.event_detection.inference.traj3d_predictor import Traj3DEventPredictor
from src.event_detection.inference.uv_predictor import UVEventPredictor

__all__ = [
    "UVEventPredictor",
    "Traj3DEventPredictor",
]
