"""Canonical typed model-I/O API for Court detection."""

from src.tasks.court_detection.model_io.adapters import (
    CourtDINOv3ExecutionBoundary,
    CourtModelExecutionBoundary,
    CourtModelIOAdapter,
)
from src.tasks.court_detection.model_io.contracts import (
    CourtDecodedPrediction,
    CourtKeypointPrediction,
    CourtLinePrediction,
    CourtLogits,
    CourtModelCall,
    CourtModelIOError,
    CourtModelSpec,
    CourtSegmentationPrediction,
    CourtTrainingCall,
    CourtTrainingResult,
)

__all__ = [
    "CourtDINOv3ExecutionBoundary",
    "CourtDecodedPrediction",
    "CourtKeypointPrediction",
    "CourtLinePrediction",
    "CourtLogits",
    "CourtModelCall",
    "CourtModelExecutionBoundary",
    "CourtModelIOAdapter",
    "CourtModelIOError",
    "CourtModelSpec",
    "CourtSegmentationPrediction",
    "CourtTrainingCall",
    "CourtTrainingResult",
]
