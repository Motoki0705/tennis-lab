"""Canonical motion contracts shared by PLCS motion sources."""

from src.tasks.plcs.motion.artifact import load_motion_clip, save_motion_clip
from src.tasks.plcs.motion.contracts import (
    COCO17_MOTION_SCHEMA_VERSION,
    Coco17MotionClip,
    MotionSourceKind,
)
from src.tasks.plcs.motion.geometry import PlacedCoco17Motion, place_motion_on_court

__all__ = [
    "COCO17_MOTION_SCHEMA_VERSION",
    "Coco17MotionClip",
    "MotionSourceKind",
    "PlacedCoco17Motion",
    "load_motion_clip",
    "place_motion_on_court",
    "save_motion_clip",
]
