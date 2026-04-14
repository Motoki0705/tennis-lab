"""Ball Localization in Court System (BLCS).

This module estimates the 3D trajectory of a tennis ball in court coordinates
from 2D ball observations and court keypoints.
"""

from src.tasks.blcs.models.blcs_model import BLCSModel

__all__ = ["BLCSModel"]
__version__ = "0.1.0"
