"""Unit tests for scene figure/axes theming."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import same_color

from src.utils.rendering.theme import (
    DARK_THEME,
    LIGHT_THEME,
    apply_axes_layout_3d,
    apply_axes_theme_3d,
    apply_figure_theme,
    resolve_theme,
)


class TestResolveTheme:
    def test_known_themes(self) -> None:
        assert resolve_theme("light") is LIGHT_THEME
        assert resolve_theme("dark") is DARK_THEME

    def test_unknown_theme_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown theme"):
            resolve_theme("sepia")

    def test_dark_theme_has_matching_court_style(self) -> None:
        assert DARK_THEME.court_style is not None
        assert LIGHT_THEME.court_style is None


class TestApplyTheme:
    def test_dark_styles_figure_and_axes(self) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        try:
            apply_figure_theme(fig, DARK_THEME)
            apply_axes_layout_3d(ax, DARK_THEME)
            apply_axes_theme_3d(ax, DARK_THEME)

            assert DARK_THEME.figure_color is not None
            assert same_color(fig.patch.get_facecolor(), DARK_THEME.figure_color)
            # Broadcast look: axes chrome removed, axes fill the figure
            # (the active position may shrink again for aspect handling).
            assert not ax._axis3don
            assert ax.get_position(original=True).bounds == (0.0, 0.0, 1.0, 1.0)
        finally:
            plt.close(fig)

    def test_light_keeps_matplotlib_defaults(self) -> None:
        fig = plt.figure()
        default_facecolor = fig.patch.get_facecolor()
        ax = fig.add_subplot(111, projection="3d")
        default_bbox = ax.get_position(original=True)
        try:
            apply_figure_theme(fig, LIGHT_THEME)
            apply_axes_layout_3d(ax, LIGHT_THEME)
            apply_axes_theme_3d(ax, LIGHT_THEME)

            assert fig.patch.get_facecolor() == default_facecolor
            assert ax._axis3don
            assert ax.get_position(original=True).bounds == default_bbox.bounds
        finally:
            plt.close(fig)

    def test_axes_theme_survives_clear(self) -> None:
        """Renderers reapply the axes theme after every ax.clear()."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        try:
            apply_axes_theme_3d(ax, DARK_THEME)
            ax.clear()
            assert ax._axis3don  # clear() resets the chrome

            apply_axes_theme_3d(ax, DARK_THEME)
            assert not ax._axis3don
        finally:
            plt.close(fig)
