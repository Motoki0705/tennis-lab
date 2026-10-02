"""Canonical BLCS model classes and discriminator construction.

Trajectory model construction lives in :mod:`src.tasks.blcs.model_io` so a
model can never be selected independently from its I/O adapter.
"""

from __future__ import annotations

from src.tasks.blcs.models.blcs_multiview_axial_model import BLCSMultiViewAxialModel
from src.tasks.blcs.models.discriminators import build_blcs_discriminator

__all__ = ["BLCSMultiViewAxialModel", "build_blcs_discriminator"]
