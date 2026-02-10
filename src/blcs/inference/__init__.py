"""BLCS inference modules."""

from src.blcs.inference.predictor import BLCSPredictor
from src.blcs.inference.multiview_predictor import BLCSMultiViewPredictor
__all__ = [
    "BLCSPredictor",
    "BLCSMultiViewPredictor"
]
