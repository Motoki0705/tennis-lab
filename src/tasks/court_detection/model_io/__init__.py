"""Canonical typed model-I/O API for court detection."""

from src.tasks.court_detection.model_io.adapters import (
    CourtKeypointModelIO,
    CourtLineModelIO,
    CourtModelIOAdapter,
    CourtSegmentationModelIO,
)
from src.tasks.court_detection.model_io.contracts import (
    CourtKeypointPrediction,
    CourtLinePrediction,
    CourtModelIOError,
    CourtModelSpec,
    CourtSegmentationPrediction,
)

__all__ = [
    "CourtKeypointModelIO",
    "CourtKeypointPrediction",
    "CourtLineModelIO",
    "CourtLinePrediction",
    "CourtModelIOAdapter",
    "CourtModelIOError",
    "CourtModelSpec",
    "CourtSegmentationModelIO",
    "CourtSegmentationPrediction",
]
