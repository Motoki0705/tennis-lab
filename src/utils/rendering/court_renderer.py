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
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.utils.schema.court import (
    CENTER_MARK_LENGTH,
    COURT_SKELETON,
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    NET_HEIGHT_CENTER,
    NET_HEIGHT_POST,
    STANDARD_COURT_CONFIG,
    court_keypoints_3d,
    net_height_at_x,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    Axes3D: TypeAlias = Any


# Color defaults
DEFAULT_COURT_COLOR: str = "#2E7D32"  # Tennis court green
DEFAULT_APRON_COLOR: str = "#1F5723"  # Darker green run-off surround
DEFAULT_LINE_COLOR: str = "white"
DEFAULT_NET_COLOR: str = "#404040"
DEFAULT_POST_COLOR: str = "#30343A"
DEFAULT_BAND_COLOR: str = "#F5F5F5"

# Default fence margin (meters)
DEFAULT_FENCE_MARGIN: float = 3.66  # Standard runback area

# 3D rendering constants
DEFAULT_3D_VIEW_MARGIN: float = 2.0  # Margin around court for limits and apron (m)
_LINE_Z_OFFSET: float = 0.01  # Lift court lines above the surface plane
_APRON_Z_OFFSET: float = -0.002  # Sink apron slightly below the court plane
_NET_BAND_HEIGHT: float = 0.06  # White band depth along the net top (m)
_NET_STRAND_SPACING_X: float = 0.5  # Vertical net strand spacing (m)
_NET_STRAND_SPACING_Z: float = 0.15  # Horizontal net strand spacing (m)


def net_top_curve(num_points: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Sample the net top cable as ``(x, z)`` arrays.

    Heights come from :func:`src.utils.schema.court.net_height_at_x`, keeping
    the rendered sag consistent with the shared court geometry definition.

    Args:
        num_points: Number of samples across the net width.

    Returns:
        Tuple of arrays with shape (num_points,): x positions and net heights.
    """
    if num_points < 2:
        raise ValueError(f"num_points must be >= 2, got {num_points}")
    x = np.linspace(-HALF_DOUBLES_WIDTH, HALF_DOUBLES_WIDTH, num_points)
    z = np.array([net_height_at_x(float(xi)) for xi in x])
    return x, z


@dataclass
class CourtStyle:
    """Style configuration for court rendering.

    Attributes:
        line_color: Color for court lines.
        line_width: Width of court lines in points.
        court_color: Background color for court surface.
        apron_color: Color for the run-off surface surrounding the court (3D).
        net_color: Color for net.
        post_color: Color for net posts (3D).
        band_color: Color for the white band along the net top (3D).
        fence_color: Color for fence boundary.
        surface_alpha: Alpha transparency for court surface.

    """

    line_color: str = DEFAULT_LINE_COLOR
    line_width: float = 2.0
    court_color: str = DEFAULT_COURT_COLOR
    apron_color: str = DEFAULT_APRON_COLOR
    net_color: str = DEFAULT_NET_COLOR
    post_color: str = DEFAULT_POST_COLOR
    band_color: str = DEFAULT_BAND_COLOR
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
        pts = court_keypoints_3d(STANDARD_COURT_CONFIG).numpy()  # (20, 3)

        segments = []

        # 1. Main court lines from COURT_SKELETON
        # Filter out net-related lines (indices >= 14) as we render net separately
        for i, j in COURT_SKELETON:
            # Skip if either keypoint is part of the net structure (14..19)
            if i >= 14 or j >= 14:
                continue

            p1 = pts[i]
            p2 = pts[j]
            segments.append(
                ((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])))
            )

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

    @staticmethod
    def _clip_segment_to_rectangle(
        start: np.ndarray,
        end: np.ndarray,
        *,
        bounds: tuple[float, float, float, float],
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """Clip a 2D segment to an axis-aligned rectangle.

        Args:
            start: Segment start point as ``(x, y)``.
            end: Segment end point as ``(x, y)``.
            bounds: Rectangle bounds as ``(x_min, x_max, y_min, y_max)``.

        Returns:
            Clipped segment endpoints, or ``None`` when the segment does not
            intersect the rectangle.

        """
        x0 = float(start[0])
        y0 = float(start[1])
        x1 = float(end[0])
        y1 = float(end[1])
        if not np.all(np.isfinite([x0, y0, x1, y1])):
            return None

        x_min, x_max, y_min, y_max = bounds
        dx = x1 - x0
        dy = y1 - y0
        p = (-dx, dx, -dy, dy)
        q = (x0 - x_min, x_max - x0, y0 - y_min, y_max - y0)

        t_enter = 0.0
        t_leave = 1.0
        epsilon = 1e-12
        for p_i, q_i in zip(p, q, strict=True):
            if abs(p_i) <= epsilon:
                if q_i < 0.0:
                    return None
                continue

            t = q_i / p_i
            if p_i < 0.0:
                if t > t_leave:
                    return None
                t_enter = max(t_enter, t)
            else:
                if t < t_enter:
                    return None
                t_leave = min(t_leave, t)

        clipped_start = (x0 + t_enter * dx, y0 + t_enter * dy)
        clipped_end = (x0 + t_leave * dx, y0 + t_leave * dy)
        return clipped_start, clipped_end

    def render_projected_2d(
        self,
        ax: Axes,
        keypoints: np.ndarray,
        visibility: np.ndarray | None = None,
        *,
        view_bounds: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
        line_color: str | None = None,
        line_width: float | None = None,
        visible_line_alpha: float = 0.8,
        partial_line_alpha: float | None = None,
        keypoint_color: str | None = None,
        keypoint_size: float = 25.0,
        keypoint_alpha: float = 0.7,
        keypoint_marker: str = "s",
        show_lines: bool = True,
        show_keypoints: bool = True,
    ) -> None:
        """Render projected court lines in normalized image coordinates.

        This method clips partially visible court lines to the viewport so that
        segments remain visible even when one endpoint falls outside the frame.

        Args:
            ax: Matplotlib axes to draw on.
            keypoints: Projected court keypoints with shape ``(N, 2)``.
            visibility: Per-keypoint visibility flags. If omitted, all points are
                treated as visible.
            view_bounds: View rectangle as ``(x_min, x_max, y_min, y_max)``.
            line_color: Override court line color.
            line_width: Override court line width.
            visible_line_alpha: Alpha used when both endpoints are visible.
            partial_line_alpha: Alpha used when only one endpoint is visible.
            keypoint_color: Override keypoint marker color.
            keypoint_size: Keypoint marker size.
            keypoint_alpha: Keypoint marker alpha.
            keypoint_marker: Keypoint marker style.
            show_lines: Whether to draw court line segments.
            show_keypoints: Whether to draw visible keypoint markers.

        """
        line_color = line_color or self.style.line_color
        line_width = line_width if line_width is not None else self.style.line_width
        keypoint_color = keypoint_color or line_color

        keypoints = np.asarray(keypoints, dtype=np.float64)
        num_keypoints = int(keypoints.shape[0])
        if visibility is None:
            visibility = np.ones(num_keypoints, dtype=bool)
        visibility = np.asarray(visibility, dtype=bool)

        if show_lines:
            for i, j in COURT_SKELETON:
                if i >= num_keypoints or j >= num_keypoints:
                    continue

                if not (visibility[i] or visibility[j]):
                    continue

                clipped = self._clip_segment_to_rectangle(
                    keypoints[i],
                    keypoints[j],
                    bounds=view_bounds,
                )
                if clipped is None:
                    continue

                alpha = visible_line_alpha
                if partial_line_alpha is not None and not (
                    visibility[i] and visibility[j]
                ):
                    alpha = partial_line_alpha

                (x0, y0), (x1, y1) = clipped
                ax.plot(
                    [x0, x1],
                    [y0, y1],
                    color=line_color,
                    linewidth=line_width,
                    alpha=alpha,
                    zorder=1,
                    solid_capstyle="round",
                )

        if not show_keypoints:
            return

        visible_keypoints = keypoints[visibility]
        if len(visible_keypoints) == 0:
            return

        ax.scatter(
            visible_keypoints[:, 0],
            visible_keypoints[:, 1],
            c=keypoint_color,
            s=keypoint_size,
            marker=keypoint_marker,
            alpha=keypoint_alpha,
            zorder=2,
        )

    def render_3d(
        self,
        ax: Axes3D,
        *,
        show_surface: bool = True,
        show_net: bool = True,
        set_limits: bool = True,
        show_apron: bool = True,
        apron_bounds: tuple[float, float, float, float] | None = None,
    ) -> None:
        """Render court in 3D.

        Args:
            ax: Matplotlib 3D axes to draw on.
            show_surface: Whether to show court surface plane.
            show_net: Whether to show net.
            set_limits: Whether to set axis limits and labels.
            show_apron: Whether to draw the run-off surface around the court.
            apron_bounds: Apron extent as ``(x_min, x_max, y_min, y_max)``.
                Defaults to the court extended by ``DEFAULT_3D_VIEW_MARGIN``
                so the apron exactly fills the default view. mplot3d does not
                clip geometry to the axes box, so pass the visible bounds when
                using custom limits.

        """
        style = self.style

        if show_surface:
            # Run-off apron slightly below the court plane so the court
            # surface wins the depth sort when viewed from above.
            if show_apron:
                if apron_bounds is None:
                    apron_bounds = (
                        -HALF_DOUBLES_WIDTH - DEFAULT_3D_VIEW_MARGIN,
                        HALF_DOUBLES_WIDTH + DEFAULT_3D_VIEW_MARGIN,
                        -HALF_LENGTH - DEFAULT_3D_VIEW_MARGIN,
                        HALF_LENGTH + DEFAULT_3D_VIEW_MARGIN,
                    )
                ap_x_min, ap_x_max, ap_y_min, ap_y_max = apron_bounds
                apron_x = np.array([[ap_x_min, ap_x_min], [ap_x_max, ap_x_max]])
                apron_y = np.array([[ap_y_min, ap_y_max], [ap_y_min, ap_y_max]])
                apron_z = np.full_like(apron_x, _APRON_Z_OFFSET)
                # 3D surfaces stay translucent (70% of surface_alpha): mplot3d
                # depth-sorts whole artists, so an opaque ground quad would
                # hide lines and players depending on the viewing angle.
                ax.plot_surface(
                    apron_x,
                    apron_y,
                    apron_z,
                    color=style.apron_color,
                    alpha=style.surface_alpha * 0.7,
                    zorder=0,
                )

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

        # Draw court lines, lifted slightly above the surface so they are not
        # swallowed by the depth sort against the surface plane.
        for (x1, y1), (x2, y2) in self.court_lines.lines:
            ax.plot(
                [x1, x2],
                [y1, y2],
                [_LINE_Z_OFFSET, _LINE_Z_OFFSET],
                color=style.line_color,
                linewidth=style.line_width,
                zorder=1,
            )

        # Draw net
        if show_net:
            self._render_net_3d(ax)

        # Set axis properties
        if set_limits:
            margin = DEFAULT_3D_VIEW_MARGIN
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.set_xlim(-HALF_DOUBLES_WIDTH - margin, HALF_DOUBLES_WIDTH + margin)
            ax.set_ylim(-HALF_LENGTH - margin, HALF_LENGTH + margin)
            ax.set_zlim(0, 4)

            # Set aspect ratio for tennis court proportions
            x_range = (HALF_DOUBLES_WIDTH + margin) * 2
            y_range = (HALF_LENGTH + margin) * 2
            z_range = 4
            ax.set_box_aspect([x_range, y_range, z_range])

    def _render_net_3d(self, ax: Axes3D) -> None:
        """Render net in 3D view: mesh strands, top band, posts, and strap.

        The net top follows :func:`net_top_curve`, i.e. the sag defined by
        ``src.utils.schema.court.net_height_at_x``. Post positions come from
        the CourtKP20 keypoints (indices 15..18).

        Args:
            ax: Matplotlib 3D axes to draw on.

        """
        style = self.style
        x_top, z_top = net_top_curve()

        # Faint net plane backing the strands.
        net_x = np.stack([x_top, x_top])
        net_y = np.zeros_like(net_x)
        net_z = np.stack([np.zeros_like(z_top), z_top])
        ax.plot_surface(
            net_x,
            net_y,
            net_z,
            color=style.net_color,
            alpha=0.15,
            zorder=2,
        )

        # Vertical strands from the ground to the sagging top cable.
        strand_xs = np.arange(
            -HALF_DOUBLES_WIDTH, HALF_DOUBLES_WIDTH + 1e-9, _NET_STRAND_SPACING_X
        )
        for x in strand_xs:
            ax.plot(
                [x, x],
                [0.0, 0.0],
                [0.0, net_height_at_x(float(x))],
                color=style.net_color,
                linewidth=0.6,
                alpha=0.55,
                zorder=2,
            )

        # Horizontal strands: full width below the centre-strap height, split
        # into the two outer sections where the sag drops below the strand.
        sag_range = NET_HEIGHT_POST - NET_HEIGHT_CENTER
        for h in np.arange(
            _NET_STRAND_SPACING_Z, NET_HEIGHT_POST, _NET_STRAND_SPACING_Z
        ):
            if h <= NET_HEIGHT_CENTER:
                spans = [(-HALF_DOUBLES_WIDTH, HALF_DOUBLES_WIDTH)]
            else:
                x_h = float(HALF_DOUBLES_WIDTH * (h - NET_HEIGHT_CENTER) / sag_range)
                spans = [(-HALF_DOUBLES_WIDTH, -x_h), (x_h, HALF_DOUBLES_WIDTH)]
            for x0, x1 in spans:
                ax.plot(
                    [x0, x1],
                    [0.0, 0.0],
                    [h, h],
                    color=style.net_color,
                    linewidth=0.6,
                    alpha=0.55,
                    zorder=2,
                )

        # White band along the top cable.
        band_z_top = np.stack([z_top - _NET_BAND_HEIGHT, z_top])
        ax.plot_surface(
            np.stack([x_top, x_top]),
            np.zeros_like(band_z_top),
            band_z_top,
            color=style.band_color,
            alpha=0.9,
            zorder=3,
        )
        ax.plot(
            x_top,
            np.zeros_like(x_top),
            z_top,
            color=style.band_color,
            linewidth=2,
            zorder=3,
        )

        # Net posts (CourtKP20 indices 15..18) and centre strap (14 -> 19).
        kp = court_keypoints_3d(STANDARD_COURT_CONFIG).numpy()
        for base_idx, top_idx in ((15, 16), (17, 18)):
            base, top = kp[base_idx], kp[top_idx]
            ax.plot(
                [base[0], top[0]],
                [base[1], top[1]],
                [base[2], top[2]],
                color=style.post_color,
                linewidth=4.0,
                zorder=3,
                solid_capstyle="round",
            )
        strap_base, strap_top = kp[14], kp[19]
        ax.plot(
            [strap_base[0], strap_top[0]],
            [strap_base[1], strap_top[1]],
            [strap_base[2], strap_top[2]],
            color=style.band_color,
            linewidth=3.0,
            zorder=3,
        )

    def get_court_keypoints_3d(self) -> NDArray[np.float32]:
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
        return np.asarray(
            court_keypoints_3d(STANDARD_COURT_CONFIG).cpu().numpy(),
            dtype=np.float32,
        )
