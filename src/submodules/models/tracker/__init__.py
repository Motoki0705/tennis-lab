"""Configurable person detector + tracker models."""

from src.submodules.models.tracker.common import TrackRequest, TrackResult
from src.submodules.models.tracker.dino_tracker import DinoPersonTracker
from src.submodules.models.tracker.yolo_tracker import YoloPersonTracker

__all__ = [
    "DinoPersonTracker",
    "TrackRequest",
    "TrackResult",
    "YoloPersonTracker",
]
