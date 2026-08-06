"""Typed inference models with a shared :class:`BaseInferenceModel` interface.

Each model exposes ``load()`` / ``unload()`` / ``predict(request) -> result``:

- :class:`YoloPersonTracker` / :class:`DinoPersonTracker` — video -> bbox tracks
- :class:`ViTPosePose2D` — video + boxes -> COCO-17 keypoints
- :class:`Hmr2FeatureExtractor` — video + boxes -> per-frame image features
- :class:`GvhmrMeshRecovery` — keypoints + boxes + features -> SMPL-X params
"""

from src.submodules.models._base.inference_model import BaseInferenceModel
from src.submodules.models.dino.person_detector import (
    DinoPersonDetector,
    PersonDetectionRequest,
    PersonDetectionResult,
)
from src.submodules.models.gvhmr.mesh_recovery import (
    GvhmrMeshRecovery,
    GvhmrRequest,
    GvhmrResult,
    SmplVertexReconstructor,
)
from src.submodules.models.hmr2.feature_extractor import (
    Hmr2FeatureExtractor,
    ImageFeatureRequest,
    ImageFeatureResult,
)
from src.submodules.models.tracker.common import TrackRequest, TrackResult
from src.submodules.models.tracker.dino_tracker import DinoPersonTracker
from src.submodules.models.tracker.yolo_tracker import YoloPersonTracker
from src.submodules.models.vitpose.pose2d import (
    Pose2DRequest,
    Pose2DResult,
    ViTPosePose2D,
)

__all__ = [
    "BaseInferenceModel",
    "DinoPersonDetector",
    "DinoPersonTracker",
    "GvhmrMeshRecovery",
    "GvhmrRequest",
    "GvhmrResult",
    "Hmr2FeatureExtractor",
    "ImageFeatureRequest",
    "ImageFeatureResult",
    "PersonDetectionRequest",
    "PersonDetectionResult",
    "Pose2DRequest",
    "Pose2DResult",
    "SmplVertexReconstructor",
    "TrackRequest",
    "TrackResult",
    "ViTPosePose2D",
    "YoloPersonTracker",
]
