"""Unit tests for reusable 3D drawing effects."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from src.utils.rendering.effects import (
    render_fading_line_3d,
    render_ground_ring,
    render_ground_shadow,
)


@pytest.fixture
def ax3d() -> Iterator[Axes3D]:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    yield ax
    plt.close(fig)


class TestRenderFadingLine3D:
    def test_draws_one_segment_per_consecutive_pair(self, ax3d: Axes3D) -> None:
        positions = np.stack(
            [np.arange(5), np.zeros(5), np.zeros(5)], axis=-1
        ).astype(np.float32)

        collection = render_fading_line_3d(ax3d, positions, color="red")

        assert isinstance(collection, Line3DCollection)
        # 3D segments are only projected to 2D paths on draw.
        ax3d.figure.canvas.draw()
        assert len(collection.get_segments()) == 4

    def test_alpha_and_width_ramp_toward_newest(self, ax3d: Axes3D) -> None:
        positions = np.stack(
            [np.arange(4), np.zeros(4), np.zeros(4)], axis=-1
        ).astype(np.float32)

        collection = render_fading_line_3d(
            ax3d,
            positions,
            color="red",
            alpha_range=(0.1, 0.9),
            linewidth_range=(1.0, 3.0),
        )

        assert collection is not None
        alphas = collection.get_colors()[:, 3]
        widths = collection.get_linewidths()
        assert alphas[0] == pytest.approx(0.1)
        assert alphas[-1] == pytest.approx(0.9)
        assert widths[0] == pytest.approx(1.0)
        assert widths[-1] == pytest.approx(3.0)

    def test_nan_point_creates_gap(self, ax3d: Axes3D) -> None:
        positions = np.stack(
            [np.arange(5), np.zeros(5), np.zeros(5)], axis=-1
        ).astype(np.float32)
        positions[2] = np.nan

        collection = render_fading_line_3d(ax3d, positions, color="red")

        # Segments (1,2) and (2,3) are dropped: only (0,1) and (3,4) remain.
        assert collection is not None
        ax3d.figure.canvas.draw()
        assert len(collection.get_segments()) == 2

    def test_returns_none_without_two_consecutive_valid_points(self, ax3d: Axes3D) -> None:
        positions: np.ndarray = np.full((4, 3), np.nan, dtype=np.float32)
        positions[1] = (1.0, 1.0, 1.0)

        assert render_fading_line_3d(ax3d, positions, color="red") is None
        assert len(ax3d.collections) == 0

    def test_invalid_shape_raises(self, ax3d: Axes3D) -> None:
        with pytest.raises(ValueError, match="positions must have shape"):
            render_fading_line_3d(ax3d, np.zeros((3, 2)), color="red")


class TestGroundArtists:
    def test_shadow_adds_filled_polygon(self, ax3d: Axes3D) -> None:
        shadow = render_ground_shadow(ax3d, (1.0, 2.0), radius=0.4)

        assert isinstance(shadow, Poly3DCollection)
        assert shadow in ax3d.collections

    def test_shadow_invalid_radius_raises(self, ax3d: Axes3D) -> None:
        with pytest.raises(ValueError, match="radius must be positive"):
            render_ground_shadow(ax3d, (0.0, 0.0), radius=0.0)

    def test_ring_adds_line_at_requested_height(self, ax3d: Axes3D) -> None:
        render_ground_ring(ax3d, (1.0, -2.0), radius=0.3, color="gold", z=0.05)

        assert len(ax3d.lines) == 1
        _, _, z_data = ax3d.lines[0].get_data_3d()
        np.testing.assert_allclose(z_data, 0.05)

    def test_ring_invalid_radius_raises(self, ax3d: Axes3D) -> None:
        with pytest.raises(ValueError, match="radius must be positive"):
            render_ground_ring(ax3d, (0.0, 0.0), radius=-1.0, color="gold")
