"""Event detection inference utilities for WASB."""

from .event_detection_predictor import (
    EventDetectionResult,
    TrajectoryEventDetector,
    load_event_detector_from_checkpoint,
)

__all__ = [
    "EventDetectionResult",
    "TrajectoryEventDetector",
    "load_event_detector_from_checkpoint",
]

