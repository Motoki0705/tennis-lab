"""Unit tests for the scene HUD overlay and top-down minimap."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.tennis_scene.rendering.hud import HudRenderer, HudStyle, MinimapRenderer

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.axes import Axes
    from mpl_toolkits.mplot3d import Axes3D

    from src.tennis_scene.io import SceneResult


@pytest.fixture
def ax3d() -> Iterator[Axes3D]:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    yield ax
    plt.close(fig)


@pytest.fixture
def ax2d() -> Iterator[Axes]:
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def _hud_text(ax: Axes3D) -> str:
    texts = [t for t in ax.texts if t.get_text()]
    assert len(texts) == 1, f"expected exactly one HUD text block, got {len(texts)}"
    return str(texts[0].get_text())


class TestHudRenderer:
    def test_full_hud_contents(self, ax3d: Axes3D) -> None:
        HudRenderer().render(
            ax3d,
            frame_idx=45,
            num_frames=90,
            fps=30.0,
            ball_speed_ms=10.0,
            bounce_count=3,
        )

        text = _hud_text(ax3d)
        assert "Frame 45/90" in text
        assert "t=  1.50s" in text
        assert "36.0 km/h" in text
        assert "Bounces 3" in text

    def test_nan_speed_shows_placeholder(self, ax3d: Axes3D) -> None:
        HudRenderer().render(
            ax3d,
            frame_idx=0,
            num_frames=10,
            fps=30.0,
            ball_speed_ms=float("nan"),
            bounce_count=0,
        )

        assert "--" in _hud_text(ax3d)

    def test_none_values_hide_lines(self, ax3d: Axes3D) -> None:
        HudRenderer().render(
            ax3d,
            frame_idx=0,
            num_frames=10,
            fps=30.0,
            ball_speed_ms=None,
            bounce_count=None,
        )

        text = _hud_text(ax3d)
        assert "km/h" not in text
        assert "Bounces" not in text

    def test_everything_disabled_draws_nothing(self, ax3d: Axes3D) -> None:
        style = HudStyle(
            show_frame_info=False, show_ball_speed=False, show_bounce_count=False
        )
        HudRenderer(style).render(
            ax3d,
            frame_idx=0,
            num_frames=10,
            fps=30.0,
            ball_speed_ms=1.0,
            bounce_count=1,
        )

        assert [t for t in ax3d.texts if t.get_text()] == []


class TestMinimapRenderer:
    def test_draws_player_and_ball_markers(self, ax2d: Axes, tiny_scene: SceneResult) -> None:
        MinimapRenderer().render(
            ax2d,
            tiny_scene,
            frame_idx=2,
            player_colors=["#FF0000"],
            bounce_frames=np.array([2], dtype=np.int64),
        )

        # Court surface patch plus scatter collections for player/bounce/ball.
        assert len(ax2d.collections) >= 3
        assert ax2d.get_xticks().size == 0

    def test_too_few_player_colors_raise(self, ax2d: Axes, tiny_scene: SceneResult) -> None:
        with pytest.raises(ValueError, match="player_colors"):
            MinimapRenderer().render(
                ax2d,
                tiny_scene,
                frame_idx=0,
                player_colors=[],
            )
