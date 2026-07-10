"""Unit tests for shared scene-style / 3D-view config parsing."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from src.tasks.base.visualization.style import (
    SceneStyleConfig,
    parse_scene_style,
    parse_view_3d,
)
from src.utils.rendering.camera_view import CAMERA_PRESETS


class TestParseSceneStyle:
    def test_absent_section_yields_defaults(self) -> None:
        assert parse_scene_style(None) == SceneStyleConfig()

    def test_parses_dictconfig_overrides(self) -> None:
        cfg = OmegaConf.create(
            {
                "theme": "dark",
                "show_shadow": False,
                "show_trail": True,
                "trail_length": 45,
                "show_hud": False,
                "show_minimap": True,
            }
        )

        style = parse_scene_style(cfg)

        assert style == SceneStyleConfig(
            theme="dark",
            show_shadow=False,
            show_trail=True,
            trail_length=45,
            show_hud=False,
            show_minimap=True,
        )

    def test_partial_mapping_keeps_defaults(self) -> None:
        style = parse_scene_style({"theme": "dark"})

        assert style.theme == "dark"
        assert style.trail_length == SceneStyleConfig().trail_length

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown visualization.style keys"):
            parse_scene_style({"them": "dark"})

    def test_unknown_theme_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown theme"):
            parse_scene_style({"theme": "sepia"})

    def test_non_positive_trail_length_raises(self) -> None:
        with pytest.raises(ValueError, match="trail_length"):
            parse_scene_style({"trail_length": 0})

    def test_non_mapping_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a mapping"):
            parse_scene_style("dark")


class TestParseView3d:
    def test_absent_section_yields_static_broadcast(self) -> None:
        controller = parse_view_3d(None)

        assert controller.mode == "static"
        assert controller.base == CAMERA_PRESETS["broadcast"]

    def test_parses_dictconfig_preset_and_mode(self) -> None:
        cfg = OmegaConf.create({"preset": "corner", "mode": "orbit", "orbit_period_s": 12.0})

        controller = parse_view_3d(cfg)

        assert controller.mode == "orbit"
        assert controller.base == CAMERA_PRESETS["corner"]
        assert controller.orbit_period_s == pytest.approx(12.0)

    def test_invalid_spec_raises(self) -> None:
        with pytest.raises(ValueError, match="requires 'preset' or both"):
            parse_view_3d({"mode": "static"})
