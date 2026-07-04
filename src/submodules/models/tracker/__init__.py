"""Person tracking model (YOLO)."""

from src.submodules.models.tracker.yolo_tracker import (
    DEFAULT_YOLO_CHECKPOINT,
    TrackRequest,
    TrackResult,
    YoloPersonTracker,
)

__all__ = [
    "DEFAULT_YOLO_CHECKPOINT",
    "TrackRequest",
    "TrackResult",
    "YoloPersonTracker",
]
