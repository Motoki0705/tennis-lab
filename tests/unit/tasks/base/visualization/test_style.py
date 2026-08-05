"""Unit tests for shared scene-style / 3D-view config parsing."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from src.tasks.base.visualization.style import (
    SceneStyleConfig,
    parse_scene_style,
    parse_view_3d,
)
from src.utils.configuration import (
    ConfigurationTypeError,
    MissingConfigurationKeyError,
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)
from src.utils.rendering.camera_view import CAMERA_PRESETS


def _style(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "theme": "dark",
        "show_shadow": True,
        "show_trail": True,
        "trail_length": 60,
        "show_hud": True,
        "show_minimap": True,
    }
    config.update(overrides)
    return config


def _view(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "preset": "broadcast",
        "elev": None,
        "azim": None,
        "zoom": None,
        "mode": "static",
        "orbit_period_s": 10.0,
        "keyframes": None,
    }
    config.update(overrides)
    return config


class TestParseSceneStyle:
    def test_absent_section_is_rejected(self) -> None:
        with pytest.raises(ConfigurationTypeError, match="visualization.style"):
            parse_scene_style(None)

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

    def test_partial_mapping_is_rejected(self) -> None:
        with pytest.raises(MissingConfigurationKeyError, match="trail_length"):
            parse_scene_style({"theme": "dark"})

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(UnknownConfigurationKeyError, match="visualization.style.them"):
            parse_scene_style(_style(them="dark"))

    def test_unknown_theme_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown theme"):
            parse_scene_style(_style(theme="sepia"))

    def test_non_positive_trail_length_raises(self) -> None:
        with pytest.raises(SemanticConfigurationError, match="trail_length"):
            parse_scene_style(_style(trail_length=0))

    def test_non_mapping_raises(self) -> None:
        with pytest.raises(ConfigurationTypeError, match="expected mapping"):
            parse_scene_style("dark")


class TestParseView3d:
    def test_absent_section_is_rejected(self) -> None:
        with pytest.raises(ConfigurationTypeError, match="visualization.view_3d"):
            parse_view_3d(None)

    def test_parses_dictconfig_preset_and_mode(self) -> None:
        cfg = OmegaConf.create(
            _view(preset="corner", mode="orbit", orbit_period_s=12.0)
        )

        controller = parse_view_3d(cfg)

        assert controller.mode == "orbit"
        assert controller.base == CAMERA_PRESETS["corner"]
        assert controller.orbit_period_s == pytest.approx(12.0)

    def test_invalid_spec_raises(self) -> None:
        with pytest.raises(ValueError, match="requires preset or both"):
            parse_view_3d(_view(preset=None))
