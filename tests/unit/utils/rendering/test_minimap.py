"""Unit tests for the plain-array top-down minimap renderer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.utils.rendering.minimap import MinimapRenderer, MinimapStyle

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.axes import Axes


@pytest.fixture
def ax2d() -> Iterator[Axes]:
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


class TestMinimapRenderer:
    def test_draws_dots_trail_and_events(self, ax2d: Axes) -> None:
        trail = np.stack(
            [
                np.linspace(-1.0, 1.0, 5, dtype=np.float32),
                np.linspace(-8.0, -6.0, 5, dtype=np.float32),
            ],
            axis=-1,
        )
        MinimapRenderer().render(
            ax2d,
            dots=[((2.0, -8.0), "#FF0000")],
            trails=[(trail, "#CCFF00")],
            trail_dots=[((1.0, -6.0), "#CCFF00")],
            event_marks_xy=np.array([[0.5, -3.0]]),
        )

        # Court surface patch plus scatter collections for dot/event/trail dot.
        assert len(ax2d.collections) >= 3
        assert ax2d.get_xticks().size == 0
        assert ax2d.get_yticks().size == 0

    def test_non_finite_inputs_are_skipped(self, ax2d: Axes) -> None:
        nan = float("nan")
        MinimapRenderer().render(
            ax2d,
            dots=[((nan, 0.0), "#FF0000")],
            trails=[(np.full((5, 2), np.nan), "#CCFF00")],
            trail_dots=[((nan, nan), "#CCFF00")],
            event_marks_xy=np.array([[nan, 0.0]]),
        )
        baseline_collections = len(ax2d.collections)

        MinimapRenderer().render(ax2d, dots=[((1.0, 1.0), "#FF0000")])

        assert len(ax2d.collections) == baseline_collections + 1

    def test_background_alpha_from_style(self, ax2d: Axes) -> None:
        MinimapRenderer(MinimapStyle(background_alpha=0.5)).render(ax2d)

        assert ax2d.patch.get_alpha() == pytest.approx(0.5)
