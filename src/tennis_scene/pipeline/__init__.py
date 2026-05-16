"""Modular tennis scene reconstruction pipeline.

This package provides an orchestration-based pipeline for 3D tennis scene
reconstruction, with each component (Court KP, GVHMR, ball detection, PLCS,
BLCS) implemented as a separate module.
"""

from src.tennis_scene.pipeline.components.ball_detection import BallDetectionModule
from src.tennis_scene.pipeline.components.blcs import BLCSModule
from src.tennis_scene.pipeline.components.court_kp import CourtKPModule
from src.tennis_scene.pipeline.components.gvhmr import GVHMRModule
from src.tennis_scene.pipeline.components.plcs import PLCSModule
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator

__all__ = [
    "TennisSceneOrchestrator",
    "CourtKPModule",
    "GVHMRModule",
    "BallDetectionModule",
    "PLCSModule",
    "BLCSModule",
]
