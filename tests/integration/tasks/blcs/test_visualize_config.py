"""Hydra compose test for the BLCS visualize config with shared style keys."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.tasks.blcs.visualization.orchestrator import (
    RuntimeConfig,
    build_runtime_config,
)
from src.utils.configuration import (
    ConfigurationTypeError,
    UnknownConfigurationKeyError,
)
from src.utils.rendering.camera_view import CAMERA_PRESETS

pytestmark = [pytest.mark.integration]

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_CONFIG_DIR = _PROJECT_ROOT / "src" / "tasks" / "blcs" / "configs"


def _build(overrides: list[str]) -> RuntimeConfig:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        cfg = compose(config_name="visualize", overrides=overrides)
    return build_runtime_config(cfg)


def test_default_config_parses_style_and_view() -> None:
    runtime = _build([])

    assert runtime.mode == "visualize"
    assert (
        runtime.scene_path
        == _PROJECT_ROOT / "data/blcs/single_object/scenes/scene_000000"
    )
    assert runtime.animation_view == "3d"
    assert runtime.style.theme == "dark"
    assert runtime.style.show_minimap is True
    assert runtime.view_3d.mode == "static"
    assert runtime.view_3d.base == CAMERA_PRESETS["broadcast"]


def test_multiview_config_uses_canonical_single_object_scene() -> None:
    runtime = _build(["visualization=multiview"])

    assert (
        runtime.scene_path
        == _PROJECT_ROOT / "data/blcs/single_object/scenes/scene_000000"
    )


def test_style_and_view_hydra_overrides() -> None:
    runtime = _build(
        [
            "visualization.style.theme=light",
            "visualization.style.show_hud=false",
            "visualization.style.trail_length=15",
            "visualization.view_3d.preset=corner",
            "visualization.view_3d.mode=orbit",
        ]
    )

    assert runtime.style.theme == "light"
    assert runtime.style.show_hud is False
    assert runtime.style.trail_length == 15
    assert runtime.view_3d.mode == "orbit"
    assert runtime.view_3d.base == CAMERA_PRESETS["corner"]


def test_style_rejects_wrong_exact_type() -> None:
    with pytest.raises(ConfigurationTypeError, match="visualization.style.show_hud"):
        _build(["visualization.style.show_hud=truthy"])


def test_removed_run_device_is_rejected() -> None:
    with pytest.raises(UnknownConfigurationKeyError, match="configuration.run"):
        _build(["+run.device=cpu"])
