"""Canonical BLCS model classes and discriminator construction.

Trajectory model construction lives in :mod:`src.tasks.blcs.model_io` so a
model can never be selected independently from its I/O adapter.
"""

from __future__ import annotations

from src.tasks.blcs.models.blcs_model import BLCSModel
from src.tasks.blcs.models.blcs_multiview_axial_model import BLCSMultiViewAxialModel
from src.tasks.blcs.models.blcs_track_query_ablation_model import (
    BLCSTrackQueryAblationModel,
)
from src.tasks.blcs.models.blcs_track_query_model import BLCSTrackQueryModel
from src.tasks.blcs.models.blcs_track_query_reference_ablation_model import (
    BLCSTrackQueryReferenceAblationModel,
)
from src.tasks.blcs.models.blcs_track_query_reference_model import (
    BLCSTrackQueryReferenceModel,
)
from src.tasks.blcs.models.discriminators import build_blcs_discriminator

__all__ = [
    "BLCSModel",
    "BLCSTrackQueryAblationModel",
    "BLCSTrackQueryModel",
    "BLCSTrackQueryReferenceAblationModel",
    "BLCSTrackQueryReferenceModel",
    "BLCSMultiViewAxialModel",
    "build_blcs_discriminator",
]
