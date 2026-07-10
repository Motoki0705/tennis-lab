"""Unit tests for the shared 3D layer (zorder) policy."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.utils.rendering.layers import SceneLayer, enable_explicit_layering


def test_layer_order_is_monotonic() -> None:
    ordered = [
        SceneLayer.SURFACE,
        SceneLayer.GROUND,
        SceneLayer.NET,
        SceneLayer.STRUCTURE,
        SceneLayer.PLAYER,
        SceneLayer.RING,
        SceneLayer.TRAIL,
        SceneLayer.MARKER,
        SceneLayer.BALL,
        SceneLayer.OVERLAY,
    ]
    assert ordered == sorted(ordered)
    assert len(set(ordered)) == len(ordered)


def test_enable_explicit_layering_disables_depth_sort() -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    try:
        assert ax.computed_zorder

        enable_explicit_layering(ax)
        assert not ax.computed_zorder

        # The flag survives clear(); reapplying per frame keeps it that way.
        ax.clear()
        enable_explicit_layering(ax)
        assert not ax.computed_zorder
    finally:
        plt.close(fig)
