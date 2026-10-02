"""Read-only ball-detection dataset catalog for the review UI.

The package exposes versioned ball frame stores as opaque scenes with dense frame positions and multi-instance ``FrameLabel`` ground
truth.  The HTTP layer lives in
``src.tasks.ball_detection.visualization.inference.service``, which both the
review and inference apps import.
"""

from .checkpoints import (
    MINIMUM_FRAMES_BY_MODEL,
    TASK_NAME,
    BallCheckpointInfo,
    BallMetricsDefaults,
    checkpoint_roots,
    describe_checkpoint,
    scan_checkpoints,
)
from .datasets import (
    SCENE_SEPARATOR,
    BallDatasetCatalog,
    BallDatasetCatalogError,
    BallDatasetSpec,
    DatasetEntry,
    SceneFrames,
    SceneMode,
    SceneRef,
    StoreSceneFrames,
    split_scene_id,
)

__all__ = [
    "MINIMUM_FRAMES_BY_MODEL",
    "SCENE_SEPARATOR",
    "TASK_NAME",
    "BallCheckpointInfo",
    "BallDatasetCatalog",
    "BallDatasetCatalogError",
    "BallDatasetSpec",
    "BallMetricsDefaults",
    "StoreSceneFrames",
    "DatasetEntry",
    "SceneFrames",
    "SceneMode",
    "SceneRef",
    "checkpoint_roots",
    "describe_checkpoint",
    "scan_checkpoints",
    "split_scene_id",
]
