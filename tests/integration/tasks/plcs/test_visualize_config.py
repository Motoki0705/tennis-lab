"""Hydra compose test for the PLCS visualize config with shared style keys."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.tasks.plcs.visualization.orchestrator import (
    RuntimeConfig,
    build_runtime_config,
)
from src.utils.configuration import (
    ConfigurationTypeError,
    UnknownConfigurationKeyError,
)
from src.utils.hydra import validate_boundary
from src.utils.rendering.camera_view import CAMERA_PRESETS

pytestmark = [pytest.mark.integration]

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_CONFIG_DIR = _PROJECT_ROOT / "src" / "tasks" / "plcs" / "configs"


def _build(overrides: list[str]) -> RuntimeConfig:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        cfg = compose(config_name="visualize", overrides=overrides)
    validate_boundary("plcs.visualize", cfg)
    return build_runtime_config(cfg)


def test_default_config_parses_style_and_view() -> None:
    runtime = _build([])

    assert runtime.mode == "visualize"
    assert (
        runtime.scene_path
        == _PROJECT_ROOT / "data/plcs/single_object/scenes/scene_000000"
    )
    assert runtime.animation_view == "3d"
    assert runtime.canonical_pose_source == "gt"
    assert runtime.style.theme == "dark"
    assert runtime.style.show_minimap is True
    assert runtime.view_3d.mode == "static"
    assert runtime.view_3d.base == CAMERA_PRESETS["broadcast"]


def test_multiview_config_uses_canonical_single_object_scene() -> None:
    runtime = _build(["visualization=multiview"])

    assert (
        runtime.scene_path
        == _PROJECT_ROOT / "data/plcs/single_object/scenes/scene_000001"
    )


def test_style_and_view_hydra_overrides() -> None:
    runtime = _build(
        [
            "visualization.style.theme=light",
            "visualization.style.show_shadow=false",
            "visualization.view_3d.preset=side",
            "visualization.canonical_pose_source=prediction",
        ]
    )

    assert runtime.canonical_pose_source == "prediction"
    assert runtime.style.theme == "light"
    assert runtime.style.show_shadow is False
    assert runtime.view_3d.base == CAMERA_PRESETS["side"]


def test_camera_view_contract_composes_independently_for_visualization() -> None:
    runtime = _build(["court_keypoints=camera_view_v2"])

    assert runtime.court_keypoint_contract.selector == "camera_view_v2"


def test_invalid_canonical_pose_source_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="visualization.canonical_pose_source must be 'gt' or 'prediction'",
    ):
        _build(["visualization.canonical_pose_source=invalid"])


def test_style_rejects_wrong_exact_type() -> None:
    with pytest.raises(
        ConfigurationTypeError,
        match="visualization.style.show_shadow",
    ):
        _build(["visualization.style.show_shadow=truthy"])


def test_view_rejects_unknown_nested_key() -> None:
    with pytest.raises(
        UnknownConfigurationKeyError,
        match="visualization.view_3d.typo",
    ):
        _build(["+visualization.view_3d.typo=1"])
