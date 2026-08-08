"""Base module for shared abstractions."""

from src.tasks.base.configuration import (
    BaseDataConfig,
    BaseRunConfig,
    BaseTrainingConfig,
    SceneVisualizationConfig,
    TrainingRuntimeConfig,
)
from src.tasks.base.data import (
    BaseDatasetWriter,
    CameraSelection,
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
