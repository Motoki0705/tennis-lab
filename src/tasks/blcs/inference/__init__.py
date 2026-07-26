"""BLCS inference modules."""

from src.tasks.blcs.inference.predictor import BLCSPredictor
from src.tasks.blcs.inference.tracking_predictor import BLCSTrackingPredictor

__all__ = [
    "BLCSPredictor",
    "BLCSTrackingPredictor",
]
