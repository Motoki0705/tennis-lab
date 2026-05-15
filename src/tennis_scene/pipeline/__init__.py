"""Modular tennis scene reconstruction pipeline.

This package provides an orchestration-based pipeline for 3D tennis scene
reconstruction, with each component (Court KP, GVHMR, WASB, PLCS, BLCS)
implemented as a separate module.
"""

from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.tennis_scene.pipeline.components.court_kp import CourtKPModule
from src.tennis_scene.pipeline.components.gvhmr import GVHMRModule
from src.tennis_scene.pipeline.components.wasb import WASBModule
from src.tennis_scene.pipeline.components.plcs import PLCSModule
from src.tennis_scene.pipeline.components.blcs import BLCSModule

__all__ = [
    "TennisSceneOrchestrator",
    "CourtKPModule",
    "GVHMRModule",
    "WASBModule",
    "PLCSModule",
    "BLCSModule",
]
