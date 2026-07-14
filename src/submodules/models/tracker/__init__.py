"""Configurable person detector + tracker models."""

from src.submodules.models.tracker.common import TrackRequest, TrackResult
from src.submodules.models.tracker.dino_tracker import DinoPersonTracker
from src.submodules.models.tracker.yolo_tracker import (
    DEFAULT_YOLO_CHECKPOINT,
    YoloPersonTracker,
)

__all__ = [
    "DEFAULT_YOLO_CHECKPOINT",
    "DinoPersonTracker",
    "TrackRequest",
    "TrackResult",
    "YoloPersonTracker",
]
