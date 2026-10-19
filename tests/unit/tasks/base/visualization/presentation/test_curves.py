"""Verify that future measurements and checkpoint markers stay hidden."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.tasks.base.visualization.presentation.curves import (
    CurveSeries,
    draw_progressive_curves,
)

pytestmark = pytest.mark.unit


def _series(name: str = "validation", style: Any = "line") -> CurveSeries:
    return CurveSeries(
        name,
        np.array([10, 20, 30], dtype=np.int64),
        np.array([8, 4, 2], dtype=np.float64),
        style,
    )


def _axes() -> Axes:
    return Figure().subplots()


@pytest.mark.parametrize("cursor,count", [(0, 0), (19, 1), (20, 2), (35, 3)])
def test_curve_and_markers_reveal_only_reached_samples(cursor: int, count: int) -> None:
    ax = _axes()
    draw_progressive_curves(
        ax,
        [_series(), _series("points", "points")],
        cursor,
        spans=[(0, 30, "train")],
        markers=[(20, 4.0, "best")],
    )
    measured = ax.lines[0]
    np.testing.assert_array_equal(measured.get_xdata(), [10, 20, 30][:count])
    np.testing.assert_array_equal(measured.get_ydata(), [8, 4, 2][:count])
    assert ax.lines[1].get_linestyle() == "None"
    assert ax.lines[1].get_marker() == "o"
    assert any(line.get_label() == "best" for line in ax.lines) == (cursor >= 20)
    np.testing.assert_array_equal(ax.lines[-1].get_xdata(), [cursor, cursor])
    assert len(ax.patches) == 1
    assert not ax.texts


def test_caller_color_cycle_is_preserved_before_first_measurement() -> None:
    for cursor in (0, 30):
        ax = _axes()
        ax.set_prop_cycle(color=["red", "blue"])
        draw_progressive_curves(ax, [_series(), _series("second")], cursor, spans=[])
        assert [line.get_color() for line in ax.lines[:2]] == ["red", "blue"]


def test_empty_measured_series_is_supported_without_fabricating_values() -> None:
    ax = _axes()
    empty = CurveSeries(
        "concept", np.array([], dtype=np.int64), np.array([], dtype=np.float64), "line"
    )
    draw_progressive_curves(ax, [empty], 0, spans=[])
    assert len(ax.lines[0].get_xdata()) == 0


@pytest.mark.parametrize(
    "steps,values,style,error",
    [
        (np.array([1], dtype=np.float32), np.array([1.0]), "line", TypeError),
        (np.array([1]), np.array([1.0], dtype=np.float32), "line", TypeError),
        (np.array([[1]]), np.array([1.0]), "line", ValueError),
        (np.array([1]), np.array([1.0, 2.0]), "line", ValueError),
        (np.array([-1]), np.array([1.0]), "line", ValueError),
        (np.array([2, 1]), np.array([1.0, 2.0]), "line", ValueError),
        (np.array([1, 1]), np.array([1.0, 2.0]), "line", ValueError),
        (np.array([1]), np.array([np.nan]), "line", ValueError),
        (np.array([1]), np.array([np.inf]), "line", ValueError),
        (np.array([1]), np.array([1.0]), "spline", ValueError),
    ],
)
def test_invalid_curve_series_is_rejected(
    steps: Any,
    values: Any,
    style: Any,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        CurveSeries("test", steps, values, style)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"cursor_step": 1.5}, TypeError),
        ({"cursor_step": True}, TypeError),
        ({"cursor_step": -1}, ValueError),
        ({"spans": [(10, 0, "reverse")]}, ValueError),
        ({"spans": [(0, 0, "empty")]}, ValueError),
        ({"spans": [(0, 1.5, "fractional")]}, ValueError),
        ({"markers": [(20, np.nan, "best")]}, ValueError),
        ({"markers": [(0.5, 1, "best")]}, ValueError),
    ],
)
def test_invalid_cursor_and_annotations_fail_before_drawing(
    kwargs: dict[str, Any],
    error: type[Exception],
) -> None:
    ax = _axes()
    arguments: dict[str, Any] = dict(cursor_step=10, spans=[], markers=[])
    arguments.update(kwargs)
    with pytest.raises(error):
        draw_progressive_curves(ax, [_series()], **arguments)
    assert not ax.lines
    assert not ax.patches
