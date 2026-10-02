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
    FamilyInfo,
    InferenceService,
    PayloadBuilder,
    PredictionRequest,
    PredictionResult,
    SceneCatalogError,
    checkpoint_mode,
)
from .web import create_app

__all__ = [
    "CheckpointInfo",
    "CheckpointMetadataError",
    "FamilyInfo",
    "InferenceService",
    "MODE_PREVIEW",
    "MODE_SINGLE",
    "PayloadBuilder",
    "PredictionRequest",
    "PredictionResult",
    "SceneCatalogError",
    "allowed_scene_families",
    "checkpoint_mode",
    "create_app",
    "describe_checkpoint",
    "load_checkpoint_config",
    "scan_checkpoints",
]
