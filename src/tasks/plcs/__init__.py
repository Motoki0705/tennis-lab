"""PLCS: Player Localization in Court System.

This module provides models and utilities for estimating player position
and rotation in tennis court coordinates from 2D pose observations.
"""

from src.plcs.inference.predictor import PLCSPredictor
from src.plcs.models.plcs_model import PLCSModel
from src.plcs.training.lightning_module import PLCSLightningModule

__all__ = [
    "PLCSModel",
    "PLCSLightningModule",
    "PLCSPredictor",
]


def __getattr__(name: str):
    """Lazy import for optional modules."""
    if name == "rendering":
        from src.plcs import rendering

        return rendering
    if name == "SceneGenerator":
        from src.plcs.generate_dataset.scene_generator import SceneGenerator

        return SceneGenerator
    if name == "MotionSampler":
        from src.plcs.generate_dataset.sampling.motion_sampler import MotionSampler

        return MotionSampler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
