"""Top-down court minimap inset from plain NumPy arrays.

:class:`MinimapRenderer` draws a 2D court with current-position dots, recent
trails, trail-head markers, and accumulated event cross-marks. All inputs are
plain arrays/colors selected and sliced by the caller — this module knows
nothing about scene types, players, balls, or bounce semantics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.utils.rendering.court_renderer import CourtRenderer

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

XY = tuple[float, float]


@dataclass(frozen=True)
class MinimapStyle:
    """Style configuration for the top-down minimap inset.

    Attributes:
        dot_size: Current-position dot size in points^2.
        dot_edge_color: Edge color of current-position dots.
        dot_edge_width: Edge width of current-position dots.
        trail_linewidth: Trail line width in points.
        trail_alpha: Trail line alpha.
        trail_dot_size: Trail-head marker size in points^2.
        event_color: Event cross-mark color.
        event_size: Event cross-mark size in points^2.
        event_linewidth: Event cross-mark line width.
        background_alpha: Alpha of the inset background patch.
    """

    dot_size: float = 45.0
    dot_edge_color: str = "white"
    dot_edge_width: float = 1.0
    trail_linewidth: float = 1.2
    trail_alpha: float = 0.7
    trail_dot_size: float = 25.0
    event_color: str = "#FFD700"
    event_size: float = 40.0
    event_linewidth: float = 1.5
    background_alpha: float = 0.85


class MinimapRenderer:
    """Render a top-down 2D court inset for the current frame."""

    def __init__(
        self,
        style: MinimapStyle | None = None,
        court_renderer: CourtRenderer | None = None,
    ) -> None:
        self.style = style or MinimapStyle()
        self.court_renderer = court_renderer or CourtRenderer()

    def render(
        self,
        ax: Axes,
        *,
        dots: Sequence[tuple[XY, str]] = (),
        trails: Sequence[tuple[NDArray[np.float32], str]] = (),
        trail_dots: Sequence[tuple[XY, str]] = (),
        event_marks_xy: NDArray[np.float32] | None = None,
    ) -> None:
        """Draw the minimap onto a 2D axis (cleared by the caller per frame).

        Args:
            ax: Target 2D axis.
            dots: Current positions as ``((x, y), color)`` pairs — e.g.
                players. Non-finite positions are skipped.
            trails: Recent tracks as ``(points (T, >=2), color)`` pairs; the
                caller slices the window. Non-finite points are dropped and a
                trail with fewer than two finite points is skipped.
            trail_dots: Trail-head markers as ``((x, y), color)`` pairs —
                e.g. the current ball. Non-finite positions are skipped.
            event_marks_xy: Accumulated event positions ``(M, >=2)`` drawn as
                cross-marks. Non-finite positions are skipped.
        """
        style = self.style
        self.court_renderer.render_2d(ax, show_surface=True, set_limits=True)

        for (x, y), color in dots:
            if not np.isfinite([x, y]).all():
                continue
            ax.scatter(
                x,
                y,
                c=color,
                s=style.dot_size,
                zorder=10,
                edgecolors=style.dot_edge_color,
                linewidths=style.dot_edge_width,
            )

        if event_marks_xy is not None:
            for mark in np.asarray(event_marks_xy, dtype=np.float64):
                if not np.isfinite(mark[:2]).all():
                    continue
                ax.scatter(
                    mark[0],
                    mark[1],
                    c=style.event_color,
                    marker="x",
                    s=style.event_size,
                    linewidths=style.event_linewidth,
                    zorder=11,
                )

        for trail, color in trails:
            pts = np.asarray(trail, dtype=np.float64)
            valid = np.isfinite(pts).all(axis=-1)
            if valid.sum() > 1:
                ax.plot(
                    pts[valid, 0],
                    pts[valid, 1],
                    color=color,
                    linewidth=style.trail_linewidth,
                    alpha=style.trail_alpha,
                    zorder=12,
                )

        for (x, y), color in trail_dots:
            if not np.isfinite([x, y]).all():
                continue
            ax.scatter(
                x,
                y,
                c=color,
                s=style.trail_dot_size,
                zorder=13,
                edgecolors="black",
                linewidths=0.5,
            )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.patch.set_alpha(style.background_alpha)
        for spine in ax.spines.values():
            spine.set_color("#888888")
