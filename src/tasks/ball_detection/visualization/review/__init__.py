"""Read-only ball-detection dataset catalog for the review UI.

The package exposes versioned ball frame stores and the optional unified web
store as opaque
scenes with dense frame positions and multi-instance ``FrameLabel`` ground
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
    DATASET_SPECS,
    SCENE_SEPARATOR,
    SPLIT_NAMES,
    BallDatasetCatalog,
    BallDatasetCatalogError,
    BallDatasetSpec,
    DatasetEntry,
    SceneFrames,
    SceneMode,
    SceneRef,
    StoreSceneFrames,
    WebSceneFrames,
    split_scene_id,
)

__all__ = [
    "DATASET_SPECS",
    "MINIMUM_FRAMES_BY_MODEL",
    "SCENE_SEPARATOR",
    "SPLIT_NAMES",
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
    "WebSceneFrames",
    "checkpoint_roots",
    "describe_checkpoint",
    "scan_checkpoints",
    "split_scene_id",
]
