"""Shared interfaces, imported lazily so CPU annotation workers do not load models."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.tasks.base.configuration import (
        ArtifactStoreConfig,
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
    from src.tasks.base.training import BaseLightningModule, ChunkRotationCallback


__all__ = [
    "ArtifactStoreConfig",
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
]


_MODULES = {
    **dict.fromkeys(
        (
            "ArtifactStoreConfig",
            "BaseDataConfig",
            "BaseRunConfig",
            "BaseTrainingConfig",
            "ChunkDataConfig",
            "CompileConfig",
            "SceneVisualizationConfig",
            "TrainingRuntimeConfig",
        ),
        "configuration",
    ),
    **dict.fromkeys(
        (
            "BaseDatasetWriter",
            "CameraSelection",
            "ChunkGenerator",
            "ChunkInfo",
            "ChunkManager",
            "ChunkState",
            "Scene",
            "SceneDataContractError",
            "SceneDatasetBase",
            "SceneDatasetConfig",
            "SceneHeader",
            "TemporalWindow",
        ),
        "data",
    ),
    "BasePredictor": "inference.predictor",
    "BaseLightningModule": "training",
    "ChunkRotationCallback": "training",
}


def __getattr__(name: str) -> Any:
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_MODULES[name]}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
