"""Regressions for the sole canonical submodule-model public API."""

from __future__ import annotations

import importlib

import src.submodules.models as models
from src.submodules.models._base.inference_model import BaseInferenceModel


def test_models_root_owns_the_documented_public_symbols() -> None:
    expected = {
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
    }
    assert set(models.__all__) == expected
    for name in expected:
        assert getattr(models, name) is not None


def test_nested_packages_do_not_reexport_root_api() -> None:
    removed_exports = {
        "src.submodules.models._base": ("BaseInferenceModel",),
        "src.submodules.models.dino": (
            "DinoPersonDetector",
            "PersonDetectionRequest",
            "PersonDetectionResult",
        ),
        "src.submodules.models.gvhmr": ("GvhmrMeshRecovery", "GvhmrRequest"),
        "src.submodules.models.hmr2": ("Hmr2FeatureExtractor", "ImageFeatureRequest"),
        "src.submodules.models.tracker": (
            "DinoPersonTracker",
            "TrackRequest",
            "TrackResult",
            "YoloPersonTracker",
        ),
        "src.submodules.models.vitpose": ("Pose2DRequest", "ViTPosePose2D"),
    }
    for module_name, names in removed_exports.items():
        module = importlib.import_module(module_name)
        for name in names:
            assert not hasattr(module, name), f"{module_name}.{name} was restored"


def test_removed_pass_through_and_yolo_helper_aliases_are_absent() -> None:
    yolo_tracker = importlib.import_module(
        "src.submodules.models.tracker.yolo_tracker"
    )
    assert "__call__" not in BaseInferenceModel.__dict__
    assert not hasattr(yolo_tracker, "_sort_tracks")
    assert not hasattr(yolo_tracker, "_build_track_tensor")
