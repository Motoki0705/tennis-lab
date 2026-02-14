"""Inference utilities for PLCS."""

from src.plcs.inference.predictor import PLCSPredictor
from src.plcs.inference.predictor_kp3d import PLCSKeypoint3DPredictor

__all__ = [
    "PLCSPredictor",
    "PLCSKeypoint3DPredictor",
]
