"""Unit tests for the generic HUD text overlay and formatting helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from src.utils.rendering.hud import (
    HudStyle,
    format_frame_clock,
    format_speed_kmh,
    render_hud_text,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mpl_toolkits.mplot3d import Axes3D


@pytest.fixture
def ax3d() -> Iterator[Axes3D]:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    yield ax
    plt.close(fig)


def _hud_text(ax: Axes3D) -> str:
    texts = [t for t in ax.texts if t.get_text()]
    assert len(texts) == 1, f"expected exactly one HUD text block, got {len(texts)}"
    return str(texts[0].get_text())


class TestFormatters:
    def test_frame_clock(self) -> None:
        assert format_frame_clock(45, 90, 30.0) == "Frame 45/90   t=  1.50s"

    def test_frame_clock_invalid_fps_raises(self) -> None:
        with pytest.raises(ValueError, match="fps must be positive"):
            format_frame_clock(0, 10, 0.0)

    def test_speed_kmh(self) -> None:
        assert format_speed_kmh(10.0) == " 36.0 km/h"

    def test_non_finite_speed_shows_placeholder(self) -> None:
        assert "--" in format_speed_kmh(float("nan"))


class TestRenderHudText:
    def test_draws_single_text_block_with_all_lines(self, ax3d: Axes3D) -> None:
        render_hud_text(
            ax3d,
            ["Frame 45/90   t=  1.50s", "Ball speed  36.0 km/h", "Bounces 3"],
            HudStyle(text_color="#E8E8E8"),
        )

        text = _hud_text(ax3d)
        assert "Frame 45/90" in text
        assert "36.0 km/h" in text
        assert "Bounces 3" in text

    def test_empty_lines_draw_nothing(self, ax3d: Axes3D) -> None:
        render_hud_text(ax3d, [], HudStyle())

        assert [t for t in ax3d.texts if t.get_text()] == []

    def test_style_is_applied(self, ax3d: Axes3D) -> None:
        render_hud_text(ax3d, ["line"], HudStyle(text_color="#FF0000", font_size=7.0))

        (text,) = [t for t in ax3d.texts if t.get_text()]
        assert text.get_color() == "#FF0000"
        assert text.get_fontsize() == pytest.approx(7.0)
