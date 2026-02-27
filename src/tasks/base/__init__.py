"""Base module for shared abstractions."""

from src.tasks.base.data import (
    BaseDatasetWriter,
    CameraSelection,
    NPZScene,
    NPZSceneDatasetBase,
    NPZSceneHeader,
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
    "NPZScene",
    "NPZSceneDatasetBase",
    "NPZSceneHeader",
    "SceneDatasetConfig",
    "TemporalWindow",
]
