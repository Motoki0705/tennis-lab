"""Unit tests for fixed publication overview geometry."""

from __future__ import annotations

import pytest

from src.synthetic_data_generation.visualization.publication.figures import (
    overview_panel_bounds,
)


def test_overview_panel_bounds_are_ordered_and_inside_canvas() -> None:
    width, height = 600, 400
    bounds = overview_panel_bounds((width, height))

    assert tuple(label for label, _ in bounds) == (
        "Court dataset",
        "BLCS dataset",
        "PLCS dataset",
        "Alignment evidence",
        "Captured cameras",
        "BLCS / PLCS cameras",
    )
    for _, (left, top, right, bottom) in bounds:
        assert 0 <= left < right <= width
        assert 0 <= top < bottom <= height

    top_row = [rectangle for _, rectangle in bounds[:3]]
    bottom_row = [rectangle for _, rectangle in bounds[3:]]
    assert all(rectangle[1] < bottom_row[0][1] for rectangle in top_row)
    assert top_row[0][0] < top_row[1][0] < top_row[2][0]


@pytest.mark.parametrize("size", [(599, 400), (600, 399), (64, 64)])
def test_overview_panel_bounds_require_minimum_canvas(size: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="at least 600x400"):
        overview_panel_bounds(size)
