"""Base data abstractions shared across tasks."""

from src.tasks.base.data.dataset_writer import BaseDatasetWriter
from src.tasks.base.data.scene_dataset import (
    CameraSelection,
    Scene,
    SceneDatasetBase,
    SceneHeader,
    SceneDatasetConfig,
    TemporalWindow,
)

__all__ = [
    "BaseDatasetWriter",
    "CameraSelection",
    "Scene",
    "SceneDatasetBase",
    "SceneHeader",
    "SceneDatasetConfig",
    "TemporalWindow",
]
