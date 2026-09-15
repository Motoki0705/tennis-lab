"""Local GPU inference UI for PLCS scenes.

Serves a browser tool that suggests checkpoints found under an outputs root,
narrows the scene families a checkpoint can consume, runs one scene window
through the model, and draws ground truth and prediction together on a freely
orbitable 3D tennis court.
"""

from .checkpoints import (
    CheckpointInfo,
    CheckpointMetadataError,
    allowed_scene_families,
    describe_checkpoint,
    load_checkpoint_config,
    scan_checkpoints,
)
from .service import (
    MODE_PREVIEW,
    MODE_SINGLE,
    MODE_TRACKING,
    FamilyInfo,
    InferenceService,
    PayloadBuilder,
    PredictionRequest,
    PredictionResult,
    SceneCatalogError,
    checkpoint_mode,
)
from .tracking import (
    SingleWindowTrackingDataset,
    TrackingSceneError,
    TrackingWindow,
    TrackMatch,
    build_tracking_batch,
    match_tracks,
)
from .web import create_app

__all__ = [
    "CheckpointInfo",
    "CheckpointMetadataError",
    "FamilyInfo",
    "InferenceService",
    "MODE_PREVIEW",
    "MODE_SINGLE",
    "MODE_TRACKING",
    "PayloadBuilder",
    "PredictionRequest",
    "PredictionResult",
    "SceneCatalogError",
    "SingleWindowTrackingDataset",
    "TrackMatch",
    "TrackingSceneError",
    "TrackingWindow",
    "allowed_scene_families",
    "build_tracking_batch",
    "checkpoint_mode",
    "create_app",
    "describe_checkpoint",
    "load_checkpoint_config",
    "match_tracks",
    "scan_checkpoints",
]
