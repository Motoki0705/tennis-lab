"""Ball trajectory renderer for 2D and 3D visualization.

This module provides rendering of ball positions and trajectories
with support for event markers (bounces, net hits, etc.).

Example:
    >>> import numpy as np
    >>> from src.utils.rendering import BallRenderer
    >>>
    >>> renderer = BallRenderer()
    >>> trajectory = np.random.randn(50, 3)  # (T, 3) positions
    >>> fig, ax = plt.subplots()
    >>> renderer.render_trajectory_2d(ax, trajectory)

"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import PathCollection
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d import Axes3D


DEFAULT_BALL_COLOR: str = "#CCFF00"  # Tennis ball yellow-green

class BallEventType(Enum):
    """Types of ball events that can be marked on trajectories."""

    BOUNCE = "bounce"
    NET_HIT = "net_hit"
    START = "start"
    END = "end"
    IMPACT = "impact"  # Racket hit


@dataclass
class BallEvent:
    """A single ball event.

    Attributes:
        event_type: Type of the event.
        frame_idx: Frame index where event occurred.
        label: Optional label for the event.

    """

    event_type: BallEventType
    frame_idx: int
    label: str | None = None


@dataclass
class BallStyle:
    """Style configuration for ball rendering.

    Attributes:
        ball_color: Color for ball markers.
        trajectory_color: Color for trajectory line.
        ball_size: Size of ball markers in points.
        trajectory_width: Width of trajectory line.
        trajectory_alpha: Alpha for trajectory line.
        use_height_colormap: Whether to color trajectory by height (Z).
        colormap: Matplotlib colormap name for height coloring.

    """

    ball_color: str = DEFAULT_BALL_COLOR
    trajectory_color: str = "#FF6B6B"
    ball_size: float = 60.0
    trajectory_width: float = 2.0
    trajectory_alpha: float = 0.8
    use_height_colormap: bool = False
    colormap: str = "coolwarm"


# Default event marker styles
EVENT_STYLES: dict[BallEventType, dict] = {
    BallEventType.BOUNCE: {
        "color": "#FFD700",  # Gold
        "marker": "o",
        "size": 120,
        "edgecolor": "black",
        "linewidth": 2,
    },
    BallEventType.NET_HIT: {
        "color": "#FF4444",  # Red
        "marker": "x",
        "size": 100,
        "edgecolor": "black",
        "linewidth": 2,
    },
    BallEventType.START: {
        "color": "#00FF00",  # Green
        "marker": "^",
        "size": 100,
        "edgecolor": "black",
        "linewidth": 1,
    },
    BallEventType.END: {
        "color": "#000000",  # Black
        "marker": "v",
        "size": 100,
        "edgecolor": "white",
        "linewidth": 1,
    },
    BallEventType.IMPACT: {
        "color": "#FF00FF",  # Magenta
        "marker": "*",
        "size": 150,
        "edgecolor": "black",
        "linewidth": 1,
    },
}


class BallRenderer:
    """Render ball positions and trajectories in 2D or 3D.

    Supports trajectory visualization with optional height-based coloring
    and event markers for bounces, net hits, etc.

    Example:
        >>> renderer = BallRenderer()
        >>> trajectory = np.random.randn(50, 3)
        >>>
        >>> # Simple trajectory
        >>> fig, ax = plt.subplots()
        >>> renderer.render_trajectory_2d(ax, trajectory)
        >>>
        >>> # With events
        >>> events = [BallEvent(BallEventType.BOUNCE, 25)]
        >>> renderer.render_trajectory_2d(ax, trajectory, events=events)

    """

    def __init__(self, style: BallStyle | None = None) -> None:
        """Initialize ball renderer.

        Args:
            style: Style configuration. If None, uses defaults.

        """
        self.style = style or BallStyle()

    def render_ball_2d(
        self,
        ax: Axes,
        position: np.ndarray,
        *,
        style_override: BallStyle | None = None,
        label: str | None = None,
        zorder: int = 10,
    ) -> PathCollection:
        """Render a single ball position in 2D.

        Args:
            ax: Matplotlib axes to draw on.
            position: Ball position (x, y) or (x, y, z).
            style_override: Override default style.
            label: Optional label for legend.
            zorder: Z-order for layering.

        Returns:
            PathCollection from scatter plot.

        """
        style = style_override or self.style
        pos = np.atleast_2d(position)

        return ax.scatter(
            pos[:, 0],
            pos[:, 1],
            c=style.ball_color,
            s=style.ball_size,
            zorder=zorder,
            label=label,
            edgecolors="black",
            linewidths=1,
        )

    def render_ball_3d(
        self,
        ax: Axes3D,
        position: np.ndarray,
        *,
        style_override: BallStyle | None = None,
        label: str | None = None,
    ) -> PathCollection:
        """Render a single ball position in 3D.

        Args:
            ax: Matplotlib 3D axes to draw on.
            position: Ball position (x, y, z).
            style_override: Override default style.
            label: Optional label for legend.

        Returns:
            PathCollection from scatter plot.

        """
        style = style_override or self.style
        pos = np.atleast_2d(position)

        return ax.scatter(
            pos[:, 0],
            pos[:, 1],
            pos[:, 2],
            c=style.ball_color,
            s=style.ball_size,
            label=label,
            edgecolors="black",
            linewidths=1,
        )

    def render_trajectory_2d(
        self,
        ax: Axes,
        positions: np.ndarray,
        *,
        events: list[BallEvent] | None = None,
        show_start_end: bool = True,
        highlight_frame: int | None = None,
        style_override: BallStyle | None = None,
    ) -> tuple[Line2D | list, PathCollection | None]:
        """Render ball trajectory in 2D (top-down view).

        Args:
            ax: Matplotlib axes to draw on.
            positions: Ball positions, shape (T, 2) or (T, 3).
            events: List of events to mark on trajectory.
            show_start_end: Whether to mark start and end points.
            highlight_frame: Frame index to highlight with larger marker.
            style_override: Override default style.

        Returns:
            Tuple of (line artist(s), highlight scatter or None).

        """
        style = style_override or self.style
        positions = np.asarray(positions)
        T = len(positions)

        # Draw trajectory line
        if style.use_height_colormap and positions.shape[1] >= 3:
            # Color by height (Z coordinate)
            lines = self._render_colored_trajectory_2d(ax, positions, style)
        else:
            (line,) = ax.plot(
                positions[:, 0],
                positions[:, 1],
                color=style.trajectory_color,
                linewidth=style.trajectory_width,
                alpha=style.trajectory_alpha,
                zorder=2,
                solid_capstyle="round",
            )
            lines = line

        # Draw trajectory points
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c=style.ball_color,
            s=style.ball_size * 0.3,
            alpha=0.5,
            zorder=3,
        )

        # Mark start and end
        if show_start_end:
            self._render_event_marker_2d(
                ax,
                positions[0],
                BallEvent(BallEventType.START, 0, "Start"),
            )
            self._render_event_marker_2d(
                ax,
                positions[-1],
                BallEvent(BallEventType.END, T - 1, "End"),
            )

        # Mark events
        if events:
            for event in events:
                if 0 <= event.frame_idx < T:
                    self._render_event_marker_2d(ax, positions[event.frame_idx], event)

        # Highlight specific frame
        highlight_scatter = None
        if highlight_frame is not None and 0 <= highlight_frame < T:
            highlight_scatter = ax.scatter(
                [positions[highlight_frame, 0]],
                [positions[highlight_frame, 1]],
                c=style.ball_color,
                s=style.ball_size * 2,
                zorder=15,
                edgecolors="white",
                linewidths=3,
            )

        return lines, highlight_scatter

    def render_trajectory_3d(
        self,
        ax: Axes3D,
        positions: np.ndarray,
        *,
        events: list[BallEvent] | None = None,
        show_start_end: bool = True,
        highlight_frame: int | None = None,
        style_override: BallStyle | None = None,
    ) -> tuple[Line2D, PathCollection | None]:
        """Render ball trajectory in 3D.

        Args:
            ax: Matplotlib 3D axes to draw on.
            positions: Ball positions, shape (T, 3).
            events: List of events to mark on trajectory.
            show_start_end: Whether to mark start and end points.
            highlight_frame: Frame index to highlight.
            style_override: Override default style.

        Returns:
            Tuple of (line artist, highlight scatter or None).

        """
        style = style_override or self.style
        positions = np.asarray(positions)
        T = len(positions)

        # Draw trajectory line
        (line,) = ax.plot(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            color=style.trajectory_color,
            linewidth=style.trajectory_width,
            alpha=style.trajectory_alpha,
            zorder=2,
        )

        # Draw trajectory points
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            c=style.ball_color,
            s=style.ball_size * 0.3,
            alpha=0.5,
            zorder=3,
        )

        # Mark start and end
        if show_start_end:
            self._render_event_marker_3d(
                ax,
                positions[0],
                BallEvent(BallEventType.START, 0, "Start"),
            )
            self._render_event_marker_3d(
                ax,
                positions[-1],
                BallEvent(BallEventType.END, T - 1, "End"),
            )

        # Mark events
        if events:
            for event in events:
                if 0 <= event.frame_idx < T:
                    self._render_event_marker_3d(ax, positions[event.frame_idx], event)

        # Highlight specific frame
        highlight_scatter = None
        if highlight_frame is not None and 0 <= highlight_frame < T:
            highlight_scatter = ax.scatter(
                [positions[highlight_frame, 0]],
                [positions[highlight_frame, 1]],
                [positions[highlight_frame, 2]],
                c=style.ball_color,
                s=style.ball_size * 2,
                zorder=15,
                edgecolors="white",
                linewidths=3,
            )

        return line, highlight_scatter

    def render_trajectory_uv(
        self,
        ax: Axes,
        uv_coords: np.ndarray,
        visibility: np.ndarray | None = None,
        *,
        events: list[BallEvent] | None = None,
        show_start_end: bool = True,
        style_override: BallStyle | None = None,
    ) -> None:
        """Render ball trajectory in UV (image) coordinates.

        Args:
            ax: Matplotlib axes to draw on.
            uv_coords: Ball UV coordinates, shape (T, 2), normalized [0, 1].
            visibility: Visibility mask, shape (T,).
            events: List of events to mark.
            show_start_end: Whether to mark start and end.
            style_override: Override default style.

        """
        style = style_override or self.style
        uv = np.asarray(uv_coords)
        T = len(uv)

        if visibility is None:
            visibility = np.ones(T, dtype=bool)
        visibility = np.asarray(visibility, dtype=bool)

        # Draw trajectory line (all points, faded for invisible)
        ax.plot(
            uv[:, 0],
            uv[:, 1],
            color=style.trajectory_color,
            linewidth=style.trajectory_width * 0.5,
            alpha=0.3,
            zorder=1,
        )

        # Draw visible points
        visible_uv = uv[visibility]
        ax.scatter(
            visible_uv[:, 0],
            visible_uv[:, 1],
            c=style.ball_color,
            s=style.ball_size * 0.5,
            alpha=0.8,
            zorder=3,
            label="Visible",
        )

        # Draw invisible points (smaller, grayed)
        invisible_uv = uv[~visibility]
        if len(invisible_uv) > 0:
            ax.scatter(
                invisible_uv[:, 0],
                invisible_uv[:, 1],
                c="gray",
                s=style.ball_size * 0.2,
                alpha=0.3,
                zorder=2,
                label="Not visible",
            )

        # Mark start and end
        if show_start_end:
            self._render_event_marker_2d(
                ax, uv[0], BallEvent(BallEventType.START, 0, "Start")
            )
            self._render_event_marker_2d(
                ax, uv[-1], BallEvent(BallEventType.END, T - 1, "End")
            )

        # Mark events
        if events:
            for event in events:
                if 0 <= event.frame_idx < T:
                    self._render_event_marker_2d(ax, uv[event.frame_idx], event)

        # Set UV coordinate space
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)  # Flip Y for image coordinates

    def _render_colored_trajectory_2d(
        self, ax: Axes, positions: np.ndarray, style: BallStyle
    ) -> list:
        """Render trajectory with height-based coloring.

        Args:
            ax: Matplotlib axes.
            positions: Ball positions with Z coordinate.
            style: Ball style.

        Returns:
            List of line segments.

        """
        import matplotlib.pyplot as plt

        z = positions[:, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
        cmap = plt.get_cmap(style.colormap)

        lines = []
        for i in range(len(positions) - 1):
            color = cmap(z_norm[i])
            (line,) = ax.plot(
                positions[i : i + 2, 0],
                positions[i : i + 2, 1],
                color=color,
                linewidth=style.trajectory_width,
                alpha=style.trajectory_alpha,
                zorder=2,
                solid_capstyle="round",
            )
            lines.append(line)
        return lines

    def _render_event_marker_2d(
        self, ax: Axes, position: np.ndarray, event: BallEvent
    ) -> PathCollection:
        """Render event marker in 2D.

        Args:
            ax: Matplotlib axes.
            position: Position to mark.
            event: Event to render.

        Returns:
            Scatter plot collection.

        """
        event_style = EVENT_STYLES.get(
            event.event_type, EVENT_STYLES[BallEventType.BOUNCE]
        )
        return ax.scatter(
            [position[0]],
            [position[1]],
            c=event_style["color"],
            s=event_style["size"],
            marker=event_style["marker"],
            edgecolors=event_style["edgecolor"],
            linewidths=event_style["linewidth"],
            zorder=10,
            label=event.label,
        )

    def _render_event_marker_3d(
        self, ax: Axes3D, position: np.ndarray, event: BallEvent
    ) -> PathCollection:
        """Render event marker in 3D.

        Args:
            ax: Matplotlib 3D axes.
            position: Position to mark.
            event: Event to render.

        Returns:
            Scatter plot collection.

        """
        event_style = EVENT_STYLES.get(
            event.event_type, EVENT_STYLES[BallEventType.BOUNCE]
        )
        return ax.scatter(
            [position[0]],
            [position[1]],
            [position[2]],
            c=event_style["color"],
            s=event_style["size"],
            marker=event_style["marker"],
            edgecolors=event_style["edgecolor"],
            linewidths=event_style["linewidth"],
            zorder=10,
            label=event.label,
        )
