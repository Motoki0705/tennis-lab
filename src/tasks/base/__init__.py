"""Base module for shared abstractions."""

from src.tasks.base.data import (
    BaseDatasetWriter,
    CameraSelection,
    ChunkGenerator,
    ChunkInfo,
    ChunkManager,
    ChunkState,
    Scene,
    SceneDatasetBase,
    SceneDatasetConfig,
    SceneHeader,
    TemporalWindow,
)
from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.base.training import BaseLightningModule, ChunkRotationCallback

__all__ = [
    "BaseDatasetWriter",
    "BaseLightningModule",
    "BasePredictor",
    "CameraSelection",
    "ChunkGenerator",
    "ChunkInfo",
    "ChunkManager",
    "ChunkRotationCallback",
    "ChunkState",
    "Scene",
    "SceneDatasetBase",
    "SceneHeader",
    "SceneDatasetConfig",
    "TemporalWindow",
]
