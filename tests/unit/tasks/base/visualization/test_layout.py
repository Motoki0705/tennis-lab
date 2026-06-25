"""Unit tests for panel-composition primitives."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.tasks.base.visualization.layout import (
    PanelStyle,
    compose_grid,
    compose_row,
    label_panel,
)

pytestmark = pytest.mark.unit


def _panel(h: int, w: int, fill: int = 100) -> np.ndarray:
    return np.full((h, w, 3), fill, dtype=np.uint8)


def test_panel_style_defaults() -> None:
    style = PanelStyle()
    assert style.tile_gap == 12
    assert style.panel_label_height == 24
    # frozen dataclass -> immutable
    with pytest.raises(FrozenInstanceError):
        style.tile_gap = 5  # type: ignore[misc]


def test_compose_row_dimensions_and_gap() -> None:
    panels = [_panel(10, 20), _panel(10, 20), _panel(10, 20)]
    out = compose_row(panels=panels, tile_gap=4, background_rgb=(0, 0, 0))
    # width = 3*20 + 2*4 = 68
    assert out.shape == (10, 68, 3)
    # gap columns between panel0 and panel1 stay background (0)
    assert (out[:, 20:24] == 0).all()
    # first panel content preserved
    assert (out[:, 0:20] == 100).all()


def test_compose_row_single_panel_no_gap() -> None:
    out = compose_row(panels=[_panel(5, 7)], tile_gap=10, background_rgb=(0, 0, 0))
    assert out.shape == (5, 7, 3)


def test_compose_row_empty_raises() -> None:
    with pytest.raises(ValueError, match="At least one panel"):
        compose_row(panels=[], tile_gap=2, background_rgb=(0, 0, 0))


def test_compose_grid_dimensions() -> None:
    row = [_panel(8, 12), _panel(8, 12)]
    grid = compose_grid(panels=[row, row], tile_gap=3, background_rgb=(0, 0, 0))
    # row width = 2*12 + 1*3 = 27 ; height = 2*8 + 1*3 = 19
    assert grid.shape == (19, 27, 3)


def test_compose_grid_empty_raises() -> None:
    with pytest.raises(ValueError, match="At least one panel"):
        compose_grid(panels=[], tile_gap=2, background_rgb=(0, 0, 0))
    with pytest.raises(ValueError, match="At least one panel"):
        compose_grid(panels=[[]], tile_gap=2, background_rgb=(0, 0, 0))


def test_label_panel_prepends_strip() -> None:
    panel = _panel(10, 30)
    labelled = label_panel(
        panel,
        text="hi",
        label_height=12,
        background_rgb=(0, 0, 0),
        text_color_rgb=(255, 255, 255),
        text_scale=0.5,
        text_thickness=1,
    )
    assert labelled.shape == (22, 30, 3)
    # original panel preserved at the bottom
    assert (labelled[12:] == 100).all()


def test_label_panel_zero_height_returns_original() -> None:
    panel = _panel(4, 4)
    out = label_panel(
        panel,
        text="x",
        label_height=0,
        background_rgb=(0, 0, 0),
        text_color_rgb=(1, 1, 1),
        text_scale=0.5,
        text_thickness=1,
    )
    assert out is panel
