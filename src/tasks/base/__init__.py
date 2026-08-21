"""Base module for shared abstractions."""

from src.tasks.base.configuration import (
    BaseDataConfig,
    BaseRunConfig,
    BaseTrainingConfig,
    ChunkDataConfig,
    CompileConfig,
    SceneVisualizationConfig,
    TrainingRuntimeConfig,
)
from src.tasks.base.data import (
    BaseDatasetWriter,
    CameraSelection,
    ChunkGenerator,
    ChunkInfo,
    ChunkManager,
    ChunkState,
    Scene,
    SceneDataContractError,
    SceneDatasetBase,
    SceneDatasetConfig,
    SceneHeader,
    TemporalWindow,
)
from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.base.training import (
    BaseLightningModule,
    ChunkRotationCallback,
    TrackingMetricConfig,
)

__all__ = [
    "BaseDataConfig",
    "BaseDatasetWriter",
    "BaseLightningModule",
    "BasePredictor",
    "BaseRunConfig",
    "BaseTrainingConfig",
    "CameraSelection",
    "ChunkGenerator",
    "ChunkInfo",
    "ChunkManager",
    "ChunkRotationCallback",
    "ChunkState",
    "ChunkDataConfig",
    "CompileConfig",
    "Scene",
    "SceneDataContractError",
    "SceneDatasetBase",
    "SceneHeader",
    "SceneVisualizationConfig",
    "SceneDatasetConfig",
    "TemporalWindow",
    "TrainingRuntimeConfig",
    "TrackingMetricConfig",
]
