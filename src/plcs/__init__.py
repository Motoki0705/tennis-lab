"""PLCS: Player Localization in Court System.

This module provides models and utilities for estimating player position
and rotation in tennis court coordinates from 2D pose observations.
"""

from src.plcs.inference.predictor import PLCSPredictor
from src.plcs.inference.predictor_kp3d import PLCSKeypoint3DPredictor
from src.plcs.models.plcs_kp3d_model import PLCSKeypoint3DModel
from src.plcs.models.plcs_model import PLCSModel
from src.plcs.training.lightning_module import PLCSLightningModule
from src.plcs.training.lightning_module_kp3d import PLCSKeypoint3DLightningModule

__all__ = [
    "PLCSModel",
    "PLCSKeypoint3DModel",
    "PLCSLightningModule",
    "PLCSKeypoint3DLightningModule",
    "PLCSPredictor",
    "PLCSKeypoint3DPredictor",
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
