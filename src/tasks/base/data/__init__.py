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
from src.tasks.base.data.track_query_reference import (
    CAMERA_ID_PADDING_VALUE,
    STABLE_CAMERA_ID_TABLE_SCHEMA_VERSION,
    ReferenceViewBatchError,
    ReferenceViewSelection,
    ReferenceViewSelectionError,
    StableCameraIdTable,
    StableCameraIdTableError,
    TrackQueryReferenceDataError,
    include_evaluation_reference_camera,
    resolve_evaluation_reference_camera_id,
    select_seeded_training_reference_camera_id,
    validate_reference_view_batch,
    validate_reference_view_index,
)

__all__ = [
    "ChunkGenerator",
    "ChunkInfo",
    "ChunkManager",
    "ChunkState",
    "CAMERA_ID_PADDING_VALUE",
    "BaseDatasetWriter",
    "LifecycleSlotAssignment",
    "CameraSelection",
    "Scene",
    "SceneDataContractError",
    "SceneDatasetBase",
    "SceneHeader",
    "SceneDatasetConfig",
    "STABLE_CAMERA_ID_TABLE_SCHEMA_VERSION",
    "TemporalWindow",
    "ReferenceViewBatchError",
    "ReferenceViewSelection",
    "ReferenceViewSelectionError",
    "StableCameraIdTable",
    "StableCameraIdTableError",
    "TrackQueryReferenceDataError",
    "build_fixed_lifecycle_assignment",
    "pack_lifecycle_slots",
    "include_evaluation_reference_camera",
    "resolve_evaluation_reference_camera_id",
    "select_seeded_training_reference_camera_id",
    "validate_reference_view_batch",
    "validate_reference_view_index",
]
