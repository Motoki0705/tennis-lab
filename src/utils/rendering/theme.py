"""Light/dark theming for matplotlib scene figures and 3D axes.

A :class:`SceneTheme` bundles the figure/axes background, text color, axes
chrome policy, and the court style that keeps contrast on that background.
Renderers resolve a theme once (:func:`resolve_theme`) and re-apply the axes
side (:func:`apply_axes_theme_3d`) after every ``ax.clear()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.utils.rendering.court_renderer import CourtStyle

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D


@dataclass(frozen=True)
class SceneTheme:
    """Look & feel of a rendered scene figure.

    Attributes:
        name: Theme identifier (``light`` / ``dark``).
        figure_color: Figure background; None keeps the matplotlib default.
        axes_color: 3D axes background; None keeps the matplotlib default.
        text_color: Color for titles, HUD text, and labels.
        hide_axes_chrome: Remove all 3D axes chrome (panes, ticks, labels)
            for a broadcast look.
        full_bleed: Let the 3D axes fill the whole figure.
        court_style: Court style tuned for this background; None means the
            :class:`~src.utils.rendering.court_renderer.CourtRenderer`
            defaults.
    """

    name: str
    figure_color: str | None
    axes_color: str | None
    text_color: str
    hide_axes_chrome: bool
    full_bleed: bool
    court_style: CourtStyle | None


LIGHT_THEME = SceneTheme(
    name="light",
    figure_color=None,
    axes_color=None,
    text_color="#222222",
    hide_axes_chrome=False,
    full_bleed=False,
    court_style=None,
)

# Brighter two-tone court that keeps contrast on the dark background.
DARK_THEME = SceneTheme(
    name="dark",
    figure_color="#101418",
    axes_color="#101418",
    text_color="#E8E8E8",
    hide_axes_chrome=True,
    full_bleed=True,
    court_style=CourtStyle(
        court_color="#4C9B57",
        apron_color="#33763D",
        net_color="#B9C0C7",
        surface_alpha=1.0,
    ),
)

_THEMES: dict[str, SceneTheme] = {
    LIGHT_THEME.name: LIGHT_THEME,
    DARK_THEME.name: DARK_THEME,
}


def resolve_theme(name: str) -> SceneTheme:
    """Look up a theme by name.

    Raises:
        ValueError: If ``name`` is not a known theme.
    """
    theme = _THEMES.get(name)
    if theme is None:
        raise ValueError(f"Unknown theme '{name}'. Available: {sorted(_THEMES)}")
    return theme


def apply_figure_theme(fig: Figure, theme: SceneTheme) -> None:
    """Apply the theme's figure-level styling (background color)."""
    if theme.figure_color is not None:
        fig.patch.set_facecolor(theme.figure_color)


def apply_axes_layout_3d(ax: Axes3D, theme: SceneTheme) -> None:
    """Let the scene fill the whole figure when the theme is full-bleed.

    Layout is figure-level state, so this is applied once at axes creation
    (unlike :func:`apply_axes_theme_3d`, which must follow every
    ``ax.clear()``).
    """
    if theme.full_bleed:
        ax.set_position((0.0, 0.0, 1.0, 1.0))


def apply_axes_theme_3d(ax: Axes3D, theme: SceneTheme) -> None:
    """Apply the theme's 3D-axes styling; call again after ``ax.clear()``."""
    if theme.axes_color is not None:
        ax.set_facecolor(theme.axes_color)
    if theme.hide_axes_chrome:
        # Broadcast look: no axes chrome at all, scene floats on the dark bg.
        ax.set_axis_off()
