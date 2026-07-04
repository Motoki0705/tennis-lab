"""Typed inference models with a shared :class:`BaseInferenceModel` interface.

Each model exposes ``load()`` / ``unload()`` / ``predict(request) -> result``:

- :class:`YoloPersonTracker` — video -> per-person bbox tracks
- :class:`ViTPosePose2D` — video + boxes -> COCO-17 keypoints
- :class:`Hmr2FeatureExtractor` — video + boxes -> per-frame image features
- :class:`GvhmrMeshRecovery` — keypoints + boxes + features -> SMPL-X params
"""

from src.submodules.models._base import BaseInferenceModel
from src.submodules.models.gvhmr import (
    GvhmrMeshRecovery,
    GvhmrRequest,
    GvhmrResult,
    SmplVertexReconstructor,
)
from src.submodules.models.hmr2 import (
    Hmr2FeatureExtractor,
    ImageFeatureRequest,
    ImageFeatureResult,
)
from src.submodules.models.tracker import TrackRequest, TrackResult, YoloPersonTracker
from src.submodules.models.vitpose import Pose2DRequest, Pose2DResult, ViTPosePose2D

__all__ = [
    "BaseInferenceModel",
    "GvhmrMeshRecovery",
    "GvhmrRequest",
    "GvhmrResult",
    "Hmr2FeatureExtractor",
    "ImageFeatureRequest",
    "ImageFeatureResult",
    "Pose2DRequest",
    "Pose2DResult",
    "SmplVertexReconstructor",
    "TrackRequest",
    "TrackResult",
    "ViTPosePose2D",
    "YoloPersonTracker",
]
