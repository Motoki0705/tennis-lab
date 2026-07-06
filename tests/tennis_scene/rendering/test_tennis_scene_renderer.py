from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from src.tennis_scene.rendering.tennis_scene_renderer import (
    TennisSceneRenderer,
    TennisSceneStyle,
)
from src.utils.schema.court import HALF_DOUBLES_WIDTH, HALF_LENGTH


def _make_renderer() -> TennisSceneRenderer:
    renderer = TennisSceneRenderer.__new__(TennisSceneRenderer)
    renderer.style = TennisSceneStyle()
    return renderer


def test_fixed_court_view_uses_full_court_bounds() -> None:
    renderer = _make_renderer()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    try:
        renderer._set_fixed_court_view(ax)

        x_half_span = float(HALF_DOUBLES_WIDTH + 2.0)
        y_half_span = float(HALF_LENGTH + 2.0)
        assert ax.get_xlim() == pytest.approx((-x_half_span, x_half_span))
        assert ax.get_ylim() == pytest.approx((-y_half_span, y_half_span))
        assert ax.get_zlim() == pytest.approx((0.0, 4.0))
    finally:
        plt.close(fig)


def test_fixed_court_view_is_stable_across_calls() -> None:
    renderer = _make_renderer()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    try:
        renderer._set_fixed_court_view(ax)
        first_xlim = ax.get_xlim()
        first_ylim = ax.get_ylim()
        first_zlim = ax.get_zlim()

        ax.set_xlim(100.0, 200.0)
        ax.set_ylim(100.0, 200.0)
        ax.set_zlim(10.0, 20.0)
        renderer._set_fixed_court_view(ax)

        assert ax.get_xlim() == pytest.approx(first_xlim)
        assert ax.get_ylim() == pytest.approx(first_ylim)
        assert ax.get_zlim() == pytest.approx(first_zlim)
    finally:
        plt.close(fig)
