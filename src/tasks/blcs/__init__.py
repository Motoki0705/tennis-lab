"""Ball Localization in Court System (BLCS).

This module estimates the 3D trajectory of a tennis ball in court coordinates
from 2D ball observations and court keypoints.
"""

from src.tasks.blcs.inference.predictor import BLCSPredictor
from src.tasks.blcs.models import build_blcs_model
from src.tasks.blcs.models.blcs_model import BLCSModel
from src.tasks.blcs.models.blcs_multiview_axial_model import BLCSMultiViewAxialModel
from src.tasks.blcs.models.blcs_multiview_model import BLCSMultiViewModel

__all__ = [
    "BLCSModel",
    "BLCSMultiViewModel",
    "BLCSMultiViewAxialModel",
    "BLCSPredictor",
    "build_blcs_model",
]
__version__ = "0.1.0"
