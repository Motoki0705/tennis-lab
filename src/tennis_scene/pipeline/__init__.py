"""Modular tennis scene reconstruction pipeline.

This package provides an orchestration-based pipeline for 3D tennis scene
reconstruction, with each component (Court KP, GVHMR, WASB, PLCS, BLCS)
implemented as a separate module.
"""

from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.tennis_scene.pipeline.court_kp import CourtKPModule
from src.tennis_scene.pipeline.gvhmr import GVHMRModule
from src.tennis_scene.pipeline.wasb import WASBModule
from src.tennis_scene.pipeline.plcs import PLCSModule
from src.tennis_scene.pipeline.blcs import BLCSModule

__all__ = [
    "TennisSceneOrchestrator",
    "CourtKPModule",
    "GVHMRModule",
    "WASBModule",
    "PLCSModule",
    "BLCSModule",
]
