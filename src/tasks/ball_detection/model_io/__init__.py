"""Canonical typed model-I/O API for ball detection."""

from src.tasks.ball_detection.model_io.adapters import BallModelIOAdapter
from src.tasks.ball_detection.model_io.contracts import (
    BallModelCall,
    BallModelInputSpec,
    BallModelIOError,
    BallPrediction,
    BallTrainingCall,
)

__all__ = [
    "BallModelCall",
    "BallModelIOAdapter",
    "BallModelIOError",
    "BallModelInputSpec",
    "BallPrediction",
    "BallTrainingCall",
]
