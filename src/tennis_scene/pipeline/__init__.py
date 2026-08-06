"""Modular tennis scene reconstruction pipeline.

This package provides an orchestration-based pipeline for 3D tennis scene
reconstruction, with each component (Court KP, GVHMR, ball detection, PLCS,
BLCS) implemented as a separate module.
"""

from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator

__all__ = ["TennisSceneOrchestrator"]
