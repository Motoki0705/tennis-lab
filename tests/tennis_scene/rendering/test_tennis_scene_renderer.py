from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.tennis_scene.rendering.tennis_scene_renderer import (
    TennisSceneRenderer,
    TennisSceneStyle,
)
from src.utils.schema.court import HALF_DOUBLES_WIDTH, HALF_LENGTH


def _make_renderer(style: TennisSceneStyle) -> TennisSceneRenderer:
    renderer = TennisSceneRenderer.__new__(TennisSceneRenderer)
    renderer.style = style
    return renderer


def test_fixed_scene_view_does_not_follow_players() -> None:
    renderer = _make_renderer(TennisSceneStyle(view_mode="fixed", fixed_view_margin=2.0))
    players_position = np.asarray(
        [
            [[0.0, 0.0, 0.0], [20.0, 5.0, 0.0]],
            [[2.0, 0.0, 0.0], [22.0, 7.0, 0.0]],
        ],
        dtype=np.float32,
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    try:
        renderer._set_scene_view(ax, players_position=players_position, frame_idx=0)
        first_xlim = ax.get_xlim()
        first_ylim = ax.get_ylim()

        renderer._set_scene_view(ax, players_position=players_position, frame_idx=1)

        assert ax.get_xlim() == pytest.approx(first_xlim)
        assert ax.get_ylim() == pytest.approx(first_ylim)
    finally:
        plt.close(fig)


def test_fixed_court_view_uses_full_court_bounds() -> None:
    margin = 2.0
    renderer = _make_renderer(TennisSceneStyle(view_mode="fixed", fixed_view_margin=margin))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    try:
        renderer._set_fixed_court_view(ax)

        x_half_span = float(HALF_DOUBLES_WIDTH + margin)
        y_half_span = float(HALF_LENGTH + margin)
        assert ax.get_xlim() == pytest.approx((-x_half_span, x_half_span))
        assert ax.get_ylim() == pytest.approx((-y_half_span, y_half_span))
        assert ax.get_zlim() == pytest.approx((0.0, 4.0))
    finally:
        plt.close(fig)


def test_player_centered_view_remains_available() -> None:
    renderer = _make_renderer(
        TennisSceneStyle(
            view_mode="player_centered",
            player_centered_half_span=(2.0, 3.0),
        )
    )
    players_position = np.asarray(
        [
            [[0.0, 0.0, 0.0], [10.0, -5.0, 0.0]],
            [[2.0, 0.0, 0.0], [14.0, -3.0, 0.0]],
        ],
        dtype=np.float32,
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    try:
        renderer._set_scene_view(ax, players_position=players_position, frame_idx=1)

        assert ax.get_xlim() == pytest.approx((10.0, 14.0))
        assert ax.get_ylim() == pytest.approx((-7.0, -1.0))
    finally:
        plt.close(fig)
