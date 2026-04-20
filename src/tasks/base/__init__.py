"""Base module for shared abstractions."""

from src.tasks.base.data import (
    BaseDatasetWriter,
    CameraSelection,
    Scene,
    SceneDatasetBase,
    SceneHeader,
    SceneDatasetConfig,
    TemporalWindow,
)
from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.base.training.lightning_module import BaseLightningModule

__all__ = [
    "BaseDatasetWriter",
    "BaseLightningModule",
    "BasePredictor",
    "CameraSelection",
    "Scene",
    "SceneDatasetBase",
    "SceneHeader",
    "SceneDatasetConfig",
    "TemporalWindow",
]
