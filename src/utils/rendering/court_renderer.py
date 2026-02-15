"""Tennis court renderer for 2D and 3D visualization.

This module provides flexible rendering of tennis courts with customizable
colors, line styles, and optional elements like fences and nets.

Example:
    >>> import matplotlib.pyplot as plt
    >>> from src.utils.rendering import CourtRenderer
    >>>
    >>> renderer = CourtRenderer()
    >>> fig, ax = plt.subplots()
    >>> renderer.render_2d(ax)
    >>> plt.show()

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from src.utils.schema.court import (
    CENTER_MARK_LENGTH,
    COURT_SKELETON,
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    HALF_SINGLES_WIDTH,
    NET_HEIGHT_CENTER,
    NET_HEIGHT_POST,
    SERVICE_LINE_DISTANCE,
    court_keypoints_3d,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from mpl_toolkits.mplot3d import Axes3D


# Color defaults
DEFAULT_COURT_COLOR: str = "#2E7D32"  # Tennis court green
DEFAULT_LINE_COLOR: str = "white"
DEFAULT_NET_COLOR: str = "#404040"

# Default fence margin (meters)
DEFAULT_FENCE_MARGIN: float = 3.66  # Standard runback area

@dataclass
class CourtStyle:
    """Style configuration for court rendering.

    Attributes:
        line_color: Color for court lines.
        line_width: Width of court lines in points.
        court_color: Background color for court surface.
        net_color: Color for net.
        fence_color: Color for fence boundary.
        surface_alpha: Alpha transparency for court surface.

    """

    line_color: str = DEFAULT_LINE_COLOR
    line_width: float = 2.0
    court_color: str = DEFAULT_COURT_COLOR
    net_color: str = DEFAULT_NET_COLOR
    fence_color: str = "#8B4513"  # Brown
    surface_alpha: float = 0.8


@dataclass
class CourtLines:
    """Court line definitions.

    Contains all line segments that make up a tennis court.
    Each line is defined as ((x1, y1), (x2, y2)).
    """

    # Computed line definitions
    lines: list[tuple[tuple[float, float], tuple[float, float]]] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        """Initialize court lines based on standard dimensions."""
        if not self.lines:
            self.lines = self._compute_lines()

    def _compute_lines(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """Compute all court line segments.

        Returns:
            List of line segments as ((x1, y1), (x2, y2)) tuples.

        """
        # Get 3D keypoints from shared geometry definition
        pts = court_keypoints_3d().numpy()  # (20, 3)

        segments = []

        # 1. Main court lines from COURT_SKELETON
        # Filter out net-related lines (indices >= 14) as we render net separately
        for i, j in COURT_SKELETON:
            # Skip if either keypoint is part of the net structure (14..19)
            if i >= 14 or j >= 14:
                continue
            
            p1 = pts[i]
            p2 = pts[j]
            segments.append(((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))))

        # 2. Add Center Marks (not in COURT_SKELETON but needed for rendering)
        # Center mark is a small line extending from baseline inward
        # Coordinates: (0, +/-HALF_LENGTH) to (0, +/-HALF_LENGTH -/+ CENTER_MARK_LENGTH)
        
        # Far center mark
        segments.append(((0.0, HALF_LENGTH), (0.0, HALF_LENGTH - CENTER_MARK_LENGTH)))
        
        # Near center mark
        segments.append(((0.0, -HALF_LENGTH), (0.0, -HALF_LENGTH + CENTER_MARK_LENGTH)))

        return segments

    @property
    def net_line(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Get net line coordinates.

        Returns:
            Net line as ((x1, y1), (x2, y2)).

        """
        return ((-HALF_DOUBLES_WIDTH, 0), (HALF_DOUBLES_WIDTH, 0))


class CourtRenderer:
    """Render tennis court in 2D or 3D.

    This renderer supports both 2D top-down views and 3D perspective views
    of a tennis court. All dimensions follow ITF standard measurements.

    Example:
        >>> renderer = CourtRenderer()
        >>> fig, ax = plt.subplots()
        >>> renderer.render_2d(ax, show_fence=True)

        >>> fig = plt.figure()
        >>> ax3d = fig.add_subplot(111, projection='3d')
        >>> renderer.render_3d(ax3d, show_net=True)

    """

    def __init__(self, style: CourtStyle | None = None) -> None:
        """Initialize court renderer.

        Args:
            style: Style configuration. If None, uses defaults.

        """
        self.style = style or CourtStyle()
        self.court_lines = CourtLines()

    def render_2d(
        self,
        ax: Axes,
        *,
        show_surface: bool = True,
        show_fence: bool = False,
        fence_margin: float = DEFAULT_FENCE_MARGIN,
        set_limits: bool = True,
    ) -> None:
        """Render court in 2D (top-down view).

        Args:
            ax: Matplotlib axes to draw on.
            show_surface: Whether to fill court surface with color.
            show_fence: Whether to show fence boundary.
            fence_margin: Margin outside court for fence (meters).
            set_limits: Whether to set axis limits and aspect ratio.

        """
        import matplotlib.pyplot as plt

        style = self.style

        # Draw court surface
        if show_surface:
            court_rect = plt.Rectangle(
                (-HALF_DOUBLES_WIDTH, -HALF_LENGTH),
                HALF_DOUBLES_WIDTH * 2,
                HALF_LENGTH * 2,
                facecolor=style.court_color,
                edgecolor="none",
                alpha=style.surface_alpha,
                zorder=0,
            )
            ax.add_patch(court_rect)

        # Draw court lines
        for (x1, y1), (x2, y2) in self.court_lines.lines:
            ax.plot(
                [x1, x2],
                [y1, y2],
                color=style.line_color,
                linewidth=style.line_width,
                zorder=1,
                solid_capstyle="round",
            )

        # Draw net (thicker line)
        (nx1, ny1), (nx2, ny2) = self.court_lines.net_line
        ax.plot(
            [nx1, nx2],
            [ny1, ny2],
            color=style.net_color,
            linewidth=style.line_width * 2,
            zorder=2,
            solid_capstyle="butt",
        )

        # Draw fence boundary
        if show_fence:
            fence_x = HALF_DOUBLES_WIDTH + fence_margin
            fence_y = HALF_LENGTH + fence_margin
            fence_rect = plt.Rectangle(
                (-fence_x, -fence_y),
                fence_x * 2,
                fence_y * 2,
                facecolor="none",
                edgecolor=style.fence_color,
                linewidth=1.5,
                linestyle="--",
                zorder=0,
            )
            ax.add_patch(fence_rect)

        # Set axis properties
        if set_limits:
            margin = fence_margin if show_fence else 2.0
            ax.set_xlim(-HALF_DOUBLES_WIDTH - margin, HALF_DOUBLES_WIDTH + margin)
            ax.set_ylim(-HALF_LENGTH - margin, HALF_LENGTH + margin)
            ax.set_aspect("equal")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")

    def render_3d(
        self,
        ax: Axes3D,
        *,
        show_surface: bool = True,
        show_net: bool = True,
        set_limits: bool = True,
    ) -> None:
        """Render court in 3D.

        Args:
            ax: Matplotlib 3D axes to draw on.
            show_surface: Whether to show court surface plane.
            show_net: Whether to show net.
            set_limits: Whether to set axis limits and labels.

        """
        style = self.style

        # Draw court surface
        if show_surface:
            court_x = np.array(
                [
                    [-HALF_DOUBLES_WIDTH, -HALF_DOUBLES_WIDTH],
                    [HALF_DOUBLES_WIDTH, HALF_DOUBLES_WIDTH],
                ]
            )
            court_y = np.array(
                [
                    [-HALF_LENGTH, HALF_LENGTH],
                    [-HALF_LENGTH, HALF_LENGTH],
                ]
            )
            court_z = np.zeros_like(court_x)
            ax.plot_surface(
                court_x,
                court_y,
                court_z,
                color=style.court_color,
                alpha=style.surface_alpha * 0.7,
                zorder=0,
            )

        # Draw court lines
        for (x1, y1), (x2, y2) in self.court_lines.lines:
            ax.plot(
                [x1, x2],
                [y1, y2],
                [0, 0],
                color=style.line_color,
                linewidth=style.line_width,
                zorder=1,
            )

        # Draw net
        if show_net:
            self._render_net_3d(ax)

        # Set axis properties
        if set_limits:
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.set_xlim(-HALF_DOUBLES_WIDTH - 2, HALF_DOUBLES_WIDTH + 2)
            ax.set_ylim(-HALF_LENGTH - 2, HALF_LENGTH + 2)
            ax.set_zlim(0, 4)

            # Set aspect ratio for tennis court proportions
            x_range = (HALF_DOUBLES_WIDTH + 2) * 2
            y_range = (HALF_LENGTH + 2) * 2
            z_range = 4
            ax.set_box_aspect([x_range, y_range, z_range])

    def _render_net_3d(self, ax: Axes3D) -> None:
        """Render net in 3D view.

        Args:
            ax: Matplotlib 3D axes to draw on.

        """
        style = self.style

        # Net as a vertical plane
        net_x = np.array(
            [
                [-HALF_DOUBLES_WIDTH, -HALF_DOUBLES_WIDTH],
                [HALF_DOUBLES_WIDTH, HALF_DOUBLES_WIDTH],
            ]
        )
        net_y = np.array([[0, 0], [0, 0]])
        net_z = np.array(
            [
                [0, NET_HEIGHT_POST],
                [0, NET_HEIGHT_POST],
            ]
        )
        ax.plot_surface(
            net_x,
            net_y,
            net_z,
            color=style.net_color,
            alpha=0.4,
            zorder=2,
        )

        # Net top line (with center sag)
        net_points = 50
        x_net = np.linspace(-HALF_DOUBLES_WIDTH, HALF_DOUBLES_WIDTH, net_points)
        y_net = np.zeros(net_points)
        # Net sags in the middle
        z_net = (
            NET_HEIGHT_POST
            - (NET_HEIGHT_POST - NET_HEIGHT_CENTER)
            * np.cos(np.pi * x_net / (2 * HALF_DOUBLES_WIDTH)) ** 2
        )
        ax.plot(x_net, y_net, z_net, color=style.net_color, linewidth=2, zorder=3)

    def get_court_keypoints_3d(self) -> np.ndarray:
        """Get 3D coordinates of standard court keypoints (CourtKP20).

        Returns CourtKP20 keypoints as defined in `src.utils.schema.court.court_keypoints_3d()`

        Keypoint indices follow the CourtKP20 specification:
        - 0..3:  far/near doubles corners
        - 4..7:  far/near singles corners
        - 8..11: service line endpoints
        - 12,13: service T (far, near)
        - 14:    net center (ground)
        - 15..18: net posts (base/top, left/right)
        - 19:    center strap top

        Returns:
            Array of shape (20, 3) containing CourtKP20 keypoint positions in meters.

        """
        return court_keypoints_3d().cpu().numpy()
