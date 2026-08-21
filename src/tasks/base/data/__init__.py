"""Base data abstractions shared across tasks."""

from src.tasks.base.data.chunk_manager import (
    ChunkGenerator,
    ChunkInfo,
    ChunkManager,
    ChunkState,
)
from src.tasks.base.data.dataset_writer import BaseDatasetWriter
from src.tasks.base.data.lifecycle_slots import (
    LifecycleSlotAssignment,
    build_fixed_lifecycle_assignment,
    pack_lifecycle_slots,
)
from src.tasks.base.data.scene_dataset import (
    CameraSelection,
    Scene,
    SceneDataContractError,
    SceneDatasetBase,
    SceneDatasetConfig,
    SceneHeader,
    TemporalWindow,
)

__all__ = [
    "ChunkGenerator",
    "ChunkInfo",
    "ChunkManager",
    "ChunkState",
    "BaseDatasetWriter",
    "LifecycleSlotAssignment",
    "CameraSelection",
    "Scene",
    "SceneDataContractError",
    "SceneDatasetBase",
    "SceneHeader",
    "SceneDatasetConfig",
    "TemporalWindow",
    "build_fixed_lifecycle_assignment",
    "pack_lifecycle_slots",
]
