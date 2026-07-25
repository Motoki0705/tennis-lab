"""Hydra compose test for the PLCS visualize config with shared style keys."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.tasks.plcs.visualization.orchestrator import build_runtime_config
from src.utils.rendering.camera_view import CAMERA_PRESETS

pytestmark = [pytest.mark.integration]

_CONFIG_DIR = Path(__file__).resolve().parents[4] / "src" / "tasks" / "plcs" / "configs"


def _build(overrides: list[str]):  # type: ignore[no-untyped-def]
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        cfg = compose(config_name="visualize", overrides=overrides)
    return build_runtime_config(cfg)


def test_default_config_parses_style_and_view() -> None:
    runtime = _build([])

    assert runtime.mode == "visualize"
    assert runtime.animation_view == "3d"
    assert runtime.canonical_pose_source == "gt"
    assert runtime.style.theme == "dark"
    assert runtime.style.show_minimap is True
    assert runtime.view_3d.mode == "static"
    assert runtime.view_3d.base == CAMERA_PRESETS["broadcast"]


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


def test_invalid_canonical_pose_source_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="visualization.canonical_pose_source must be 'gt' or 'prediction'",
    ):
        _build(["visualization.canonical_pose_source=invalid"])
