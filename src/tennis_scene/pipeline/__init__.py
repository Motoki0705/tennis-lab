"""Modular tennis scene reconstruction pipeline.

This package provides an orchestration-based pipeline for 3D tennis scene
reconstruction, with each component (Court KP, GVHMR, WASB, Trajectory,
Event-UV, PLCS, BLCS, Event-3D)
implemented as a separate module.
"""

from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.tennis_scene.pipeline.components.event_3d import Event3DModule
from src.tennis_scene.pipeline.components.event_uv import EventUVModule
from src.tennis_scene.pipeline.components.court_kp import CourtKPModule
from src.tennis_scene.pipeline.components.gvhmr import GVHMRModule
from src.tennis_scene.pipeline.components.wasb import WASBModule
from src.tennis_scene.pipeline.components.plcs import PLCSModule
from src.tennis_scene.pipeline.components.blcs import BLCSModule
from src.tennis_scene.pipeline.components.trajectory import TrajectoryModule

__all__ = [
    "TennisSceneOrchestrator",
    "CourtKPModule",
    "GVHMRModule",
    "WASBModule",
    "TrajectoryModule",
    "EventUVModule",
    "PLCSModule",
    "BLCSModule",
    "Event3DModule",
]
