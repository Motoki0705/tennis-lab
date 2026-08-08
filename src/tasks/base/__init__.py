"""Base module for shared abstractions."""

from src.tasks.base.configuration import (
    BaseDataConfig,
    BaseRunConfig,
    BaseTrainingConfig,
    SceneVisualizationConfig,
    TrainingRuntimeConfig,
)
from src.tasks.base.data import CanonicalDataset
from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.base.training import (
    BaseLightningModule,
    TrackingMetricConfig,
)

__all__ = [
    "BaseDataConfig",
    "BaseLightningModule",
    "BasePredictor",
    "BaseRunConfig",
    "BaseTrainingConfig",
    "CanonicalDataset",
    "SceneVisualizationConfig",
    "TrainingRuntimeConfig",
    "TrackingMetricConfig",
]
