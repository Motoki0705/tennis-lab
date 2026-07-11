"""Unit tests for 3D court rendering geometry."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from mpl_toolkits.mplot3d import Axes3D

from src.utils.rendering.court_renderer import CourtRenderer, net_top_curve
from src.utils.schema.court import (
    HALF_DOUBLES_WIDTH,
    NET_HEIGHT_CENTER,
    NET_HEIGHT_POST,
    NET_POST_OFFSET_X,
    net_height_at_x,
)


@pytest.fixture
def ax3d() -> Iterator[Axes3D]:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    yield ax
    plt.close(fig)


class TestNetTopCurve:
    def test_matches_schema_net_height(self) -> None:
        x, z = net_top_curve(num_points=25)

        expected = [net_height_at_x(float(xi)) for xi in x]
        np.testing.assert_allclose(z, expected)
        assert z[0] == pytest.approx(NET_HEIGHT_POST)
        assert z[-1] == pytest.approx(NET_HEIGHT_POST)

    def test_center_is_strap_height(self) -> None:
        _, z = net_top_curve(num_points=21)
        assert z[10] == pytest.approx(NET_HEIGHT_CENTER)

    def test_too_few_points_raise(self) -> None:
        with pytest.raises(ValueError, match="num_points"):
            net_top_curve(num_points=1)


class TestRender3D:
    def test_apron_adds_one_surface(self, ax3d: Axes3D) -> None:
        fig2 = plt.figure()
        ax_no_apron = fig2.add_subplot(111, projection="3d")
        try:
            CourtRenderer().render_3d(ax3d, show_apron=True)
            CourtRenderer().render_3d(ax_no_apron, show_apron=False)
            assert len(ax3d.collections) == len(ax_no_apron.collections) + 1
        finally:
            plt.close(fig2)

    def test_court_lines_are_lifted_above_surface(self, ax3d: Axes3D) -> None:
        CourtRenderer().render_3d(ax3d, show_net=False)

        assert len(ax3d.lines) > 0
        for line in ax3d.lines:
            _, _, z = line.get_data_3d()
            np.testing.assert_allclose(z, 0.01)

    def test_net_posts_reach_post_height_at_post_positions(self, ax3d: Axes3D) -> None:
        CourtRenderer().render_3d(ax3d, show_net=True)

        post_x = HALF_DOUBLES_WIDTH + NET_POST_OFFSET_X
        found_posts = 0
        for line in ax3d.lines:
            x, y, z = line.get_data_3d()
            if (
                len(x) == 2
                and abs(abs(x[0]) - post_x) < 1e-6
                and np.allclose(y, 0.0)
                and z.max() == pytest.approx(NET_HEIGHT_POST)
            ):
                found_posts += 1
        assert found_posts == 2

    def test_render_with_custom_apron_bounds_smoke(self, ax3d: Axes3D) -> None:
        CourtRenderer().render_3d(
            ax3d, show_net=False, apron_bounds=(-7.0, 7.0, -13.0, 13.0)
        )

        assert len(ax3d.collections) == 2  # apron + court surfaces
