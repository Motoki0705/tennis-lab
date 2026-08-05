"""BLCS scene renderer for ball trajectory visualization.

This module provides complete scene rendering for BLCS (Ball Location from
Court keypoints and Skeleton) data, combining court and ball trajectory
visualization. 3D views build on the shared rich-rendering primitives in
``src.utils.rendering`` (theme, layers, camera, effects, HUD, minimap);
this renderer owns only the BLCS-specific parts: scene-dict access, event
extraction from metadata, and HUD line selection.

Example:
    >>> from src.tasks.blcs.visualization.rendering import BLCSSceneRenderer
    >>>
    >>> renderer = BLCSSceneRenderer()
    >>> fig = renderer.render_multi_view(scene_data, frame_idx=0)
    >>> plt.show()

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from src.tasks.base.visualization.style import SceneStyleConfig
from src.utils.rendering.ball_renderer import (
    BallEvent,
    BallEventType,
    BallRenderer,
    BallStyle,
)
from src.utils.rendering.camera_view import CameraController, apply_scene_camera
from src.utils.rendering.court_renderer import CourtRenderer
from src.utils.rendering.effects import (
    render_fading_line_3d,
    render_ground_shadow,
    render_impact_ring,
)
from src.utils.rendering.hud import (
    HudStyle,
    format_frame_clock,
    format_speed_kmh,
    render_hud_text,
)
from src.utils.rendering.layers import SceneLayer, enable_explicit_layering
from src.utils.rendering.minimap import MinimapRenderer
from src.utils.rendering.theme import (
    apply_axes_layout_3d,
    apply_axes_theme_3d,
    apply_figure_theme,
    resolve_theme,
)
from src.utils.rendering.trajectory_analysis import compute_speeds, detect_bounces
from src.utils.schema.court import HALF_DOUBLES_WIDTH, HALF_LENGTH

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    Axes3D: TypeAlias = Any

logger = logging.getLogger(__name__)

_VIEW_MARGIN = 2.0
_VIEW_Z_LIMIT = 4.0
_BALL_SHADOW_RADIUS = 0.08
_BOUNCE_RING_COLOR = "#FFD700"
_BOUNCE_MARKER_DURATION_S = 1.5
_MINIMAP_TRAIL_FRAMES = 30
_GT_COLOR = "green"
_PRED_COLOR = "red"
_BALL_COLORS = ("#CCFF00", "#00D4FF", "#FF5CA8", "#FFB000")

# Minimap inset rectangle in figure coordinates (left, bottom, width, height).
_MINIMAP_RECT = (0.76, 0.04, 0.21, 0.30)


def extract_ball_events(meta: dict[str, Any]) -> list[BallEvent]:
    """Extract ball events from BLCS scene metadata.

    Args:
        meta: Scene metadata dictionary with a ``shots`` list.

    Returns:
        List of BallEvent objects.
    """
    events = []

    shots = meta.get("shots", [])
    for i, shot in enumerate(shots):
        shot_idx = shot.get("shot_index", i)

        if shot.get("t_start", -1) >= 0 and i > 0:
            events.append(
                BallEvent(
                    BallEventType.SHOT_BOUNDARY,
                    shot["t_start"],
                    f"Shot {shot_idx + 1} start",
                )
            )

        if shot.get("t_bounce1", -1) >= 0:
            events.append(
                BallEvent(
                    BallEventType.BOUNCE,
                    shot["t_bounce1"],
                    f"S{shot_idx + 1} Bounce 1",
                )
            )
        if shot.get("t_bounce2", -1) >= 0:
            events.append(
                BallEvent(
                    BallEventType.BOUNCE,
                    shot["t_bounce2"],
                    f"S{shot_idx + 1} Bounce 2",
                )
            )

        if shot.get("t_net", -1) >= 0:
            events.append(
                BallEvent(
                    BallEventType.NET_HIT,
                    shot["t_net"],
                    f"S{shot_idx + 1} Net hit",
                )
            )

    return events


def resolve_bounce_frames(
    positions: NDArray[np.float32],
    events: list[BallEvent] | None,
) -> NDArray[np.int64]:
    """Bounce frame indices for a trajectory, preferring event metadata.

    Bounce events from scene metadata are authoritative; only when the event
    list carries no bounces does this fall back to detecting them from the
    trajectory, so the same bounce is never reported twice.
    """
    if events:
        frames = sorted(
            e.frame_idx for e in events if e.event_type is BallEventType.BOUNCE
        )
        if frames:
            return np.asarray(frames, dtype=np.int64)
    logger.info("No bounce events in metadata; falling back to detect_bounces().")
    return detect_bounces(positions)


def split_ball_tracks(scene: dict[str, Any]) -> list[NDArray[np.float32]]:
    """Return active ``(T, 3)`` trajectories from a single/multi-ball scene."""
    positions = np.asarray(scene["ball_pos_world"], dtype=np.float32)
    if positions.ndim == 2:
        if positions.shape[1] != 3:
            raise ValueError(
                f"Expected ball positions shaped (T, 3), got {positions.shape}."
            )
        return [positions]
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError(
            f"Expected ball positions shaped (T, Q, 3), got {positions.shape}."
        )
    raw_num_balls = scene["num_balls"] if "num_balls" in scene else positions.shape[1]
    if type(raw_num_balls) is not int:
        raise TypeError("scene.num_balls must be exactly int when present.")
    num_balls = raw_num_balls
    if not 1 <= num_balls <= positions.shape[1]:
        raise ValueError(
            f"num_balls must be within [1, {positions.shape[1]}], got {num_balls}."
        )
    return [positions[:, index] for index in range(num_balls)]


def extract_ball_track_events(meta: dict[str, Any], ball_index: int) -> list[BallEvent]:
    """Extract events for one ball from either single- or multi-ball metadata."""
    shots = meta.get("shots", [])
    if shots and isinstance(shots[0], dict) and "shots" in shots[0]:
        for record in shots:
            if int(record.get("ball_index", -1)) == ball_index:
                return extract_ball_events({"shots": record.get("shots", [])})
        return []
    return extract_ball_events(meta) if ball_index == 0 else []


class BLCSSceneRenderer:
    """Render complete BLCS scenes with court and ball trajectories.

    Combines court rendering and ball trajectory rendering for comprehensive
    scene visualization in 2D, 3D, and camera UV views.

    Example:
        >>> renderer = BLCSSceneRenderer()
        >>> fig, ax = renderer.render_3d_view(scene)
        >>> fig = renderer.render_multi_view(scene, frame_idx=10)
        >>> anim = renderer.create_animation(scene, view="2d")

    """

    def __init__(
        self,
        *,
        style: SceneStyleConfig,
        court_renderer: CourtRenderer | None = None,
        ball_renderer: BallRenderer | None = None,
        camera: CameraController | None = None,
    ) -> None:
        """Initialize BLCS scene renderer.

        Args:
            court_renderer: Court renderer instance. If None, creates one
                matching the theme.
            ball_renderer: Ball renderer. If None, creates default.
            style: Shared scene-style settings (theme, shadows, trails, HUD,
                minimap) applied to the 3D views.
            camera: 3D viewpoint controller. If None, uses the static
                broadcast preset.

        """
        self.style = style
        self.theme = resolve_theme(self.style.theme)
        self.camera = camera or CameraController("broadcast")
        self.court_renderer = court_renderer or CourtRenderer(self.theme.court_style)
        self.ball_renderer = ball_renderer or BallRenderer()
        self.hud_style = HudStyle(text_color=self.theme.text_color)
        self.minimap_renderer = MinimapRenderer()

    def _get_display_title(self, meta: dict[str, Any]) -> str:
        """Generate title string for scene visualization.

        Args:
            meta: Scene metadata dictionary.

        Returns:
            Formatted title string.

        """
        scene_id = meta.get("scene_id", "Unknown")

        rally_len = meta.get("rally_length", 0)
        end_reason = meta.get("end_reason", "Unknown")
        return f"Rally: {scene_id} | {rally_len} shots | End: {end_reason}"

    def render_3d_view(
        self,
        scene: dict[str, Any],
        frame_idx: int = -1,
        *,
        figsize: tuple[float, float] = (12, 8),
        ax: Axes3D | None = None,
    ) -> tuple[Figure | None, Axes3D]:
        """Render 3D view of the full ball trajectory.

        Args:
            scene: BLCS scene dictionary with 'ball_pos_world' and 'meta'.
            frame_idx: Frame to highlight (-1 for none).
            figsize: Figure size if creating new figure.
            ax: Existing 3D axes to draw on. If None, creates new figure.

        Returns:
            Tuple of (figure or None if ax provided, 3D axes).

        """
        fig = None
        if ax is None:
            fig = plt.figure(figsize=figsize)
            apply_figure_theme(fig, self.theme)
            ax = fig.add_subplot(111, projection="3d")

        self._setup_3d_axes(ax)

        # Get ball trajectories
        tracks = split_ball_tracks(scene)
        meta = scene["meta"]

        highlight = frame_idx if frame_idx >= 0 else None
        for index, positions in enumerate(tracks):
            color = _BALL_COLORS[index % len(_BALL_COLORS)]
            self.ball_renderer.render_trajectory_3d(
                ax,
                positions,
                events=extract_ball_track_events(meta, index),
                highlight_frame=highlight,
                style_override=BallStyle(
                    ball_color=color,
                    trajectory_color=color,
                ),
            )

        apply_scene_camera(
            ax, self.camera.base, margin=_VIEW_MARGIN, z_limit=_VIEW_Z_LIMIT
        )
        if self.theme.name != "dark":
            ax.set_title(self._get_display_title(meta), color=self.theme.text_color)

        return fig, ax

    def render_2d_topdown(
        self,
        scene: dict[str, Any],
        frame_idx: int = -1,
        *,
        figsize: tuple[float, float] = (8, 12),
        ax: Axes | None = None,
        use_height_colormap: bool = True,
    ) -> tuple[Figure | None, Axes]:
        """Render 2D top-down view of ball trajectory.

        Args:
            scene: BLCS scene dictionary.
            frame_idx: Frame to highlight (-1 for none).
            figsize: Figure size if creating new figure.
            ax: Existing axes to draw on. If None, creates new figure.
            use_height_colormap: Whether to color trajectory by ball height.

        Returns:
            Tuple of (figure or None if ax provided, axes).

        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Render court
        self.court_renderer.render_2d(ax, show_fence=True)

        tracks = split_ball_tracks(scene)
        meta = scene["meta"]
        highlight = frame_idx if frame_idx >= 0 else None
        for index, positions in enumerate(tracks):
            color = _BALL_COLORS[index % len(_BALL_COLORS)]
            self.ball_renderer.render_trajectory_2d(
                ax,
                positions,
                events=extract_ball_track_events(meta, index),
                highlight_frame=highlight,
                style_override=BallStyle(
                    ball_color=color,
                    trajectory_color=color,
                    use_height_colormap=use_height_colormap,
                ),
            )

        # Title
        title = self._get_display_title(meta)
        ax.set_title(f"Top-down: {title}")

        return fig, ax

    def render_camera_view(
        self,
        scene: dict[str, Any],
        camera_idx: int = 0,
        frame_idx: int = -1,
        *,
        figsize: tuple[float, float] = (12, 8),
        ax: Axes | None = None,
    ) -> tuple[Figure | None, Axes]:
        """Render camera view with 2D ball projection.

        Args:
            scene: BLCS scene dictionary.
            camera_idx: Camera index.
            frame_idx: Frame to highlight (-1 for none).
            figsize: Figure size if creating new figure.
            ax: Existing axes to draw on.

        Returns:
            Tuple of (figure or None, axes).

        Raises:
            ValueError: If camera_idx is out of range.

        """
        num_cameras = scene["num_cameras"]
        if camera_idx >= num_cameras:
            raise ValueError(f"Camera {camera_idx} out of range (0-{num_cameras - 1})")

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        meta = scene["meta"]
        cam = scene["cameras"][camera_idx]
        ball_uv = cam["ball_uv"]
        ball_vis = cam["ball_visible"]
        court_uv = cam["court_kp_uv"]
        court_vis = cam["court_kp_visible"]

        # Set up UV coordinate space
        ax.set_facecolor("#1a1a1a")
        self.court_renderer.render_projected_2d(
            ax,
            court_uv,
            court_vis,
            line_color="lime",
            line_width=1.5,
            visible_line_alpha=0.8,
            keypoint_color="lime",
            keypoint_size=50.0,
            keypoint_alpha=0.7,
            keypoint_marker="s",
        )

        uv_tracks = (
            [ball_uv]
            if ball_uv.ndim == 2
            else [
                ball_uv[:, index]
                for index in range(int(scene.get("num_balls", ball_uv.shape[1])))
            ]
        )
        visibility_tracks = (
            [ball_vis]
            if ball_vis.ndim == 1
            else [ball_vis[:, index] for index in range(len(uv_tracks))]
        )
        for index, (uv_track, visibility) in enumerate(
            zip(uv_tracks, visibility_tracks, strict=True)
        ):
            color = _BALL_COLORS[index % len(_BALL_COLORS)]
            self.ball_renderer.render_trajectory_uv(
                ax,
                uv_track,
                visibility=visibility.astype(bool),
                events=extract_ball_track_events(meta, index),
                style_override=BallStyle(
                    ball_color=color,
                    trajectory_color=color,
                ),
            )
            if frame_idx >= 0 and frame_idx < len(uv_track):
                ax.scatter(
                    [uv_track[frame_idx, 0]],
                    [uv_track[frame_idx, 1]],
                    c=color,
                    s=200,
                    marker="o",
                    edgecolors="white",
                    linewidths=3,
                    zorder=15,
                    label=f"Ball {index + 1}",
                )

        # Title
        ax.set_title(
            f"Camera {camera_idx} | Visibility: {cam['ball_visibility_ratio']:.1%}"
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("U (normalized)")
        ax.set_ylabel("V (normalized)")
        ax.grid(True, alpha=0.3)

        return fig, ax

    def render_multi_view(
        self,
        scene: dict[str, Any],
        frame_idx: int = -1,
        *,
        figsize: tuple[float, float] = (16, 10),
    ) -> Figure:
        """Render multiple views in a single figure.

        Shows 3D view, 2D top-down, and up to 2 camera views.

        Args:
            scene: BLCS scene dictionary.
            frame_idx: Frame to highlight (-1 for none).
            figsize: Figure size.

        Returns:
            Figure with all views.

        """
        fig: Figure = plt.figure(figsize=figsize)

        # 3D view
        ax1 = fig.add_subplot(2, 2, 1, projection="3d")
        self.render_3d_view(scene, frame_idx, ax=ax1)

        # 2D top-down view
        ax2 = fig.add_subplot(2, 2, 2)
        self.render_2d_topdown(scene, frame_idx, ax=ax2)

        # Camera view 0
        ax3 = fig.add_subplot(2, 2, 3)
        self.render_camera_view(scene, 0, frame_idx, ax=ax3)

        # Camera view 1 (if available)
        ax4 = fig.add_subplot(2, 2, 4)
        if scene["num_cameras"] > 1:
            self.render_camera_view(scene, 1, frame_idx, ax=ax4)
        else:
            ax4.text(
                0.5,
                0.5,
                "No second camera",
                ha="center",
                va="center",
                fontsize=14,
                transform=ax4.transAxes,
            )
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)

        fig.suptitle(
            f"BLCS Scene: {scene['meta']['scene_id']}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        return fig

    def _setup_3d_axes(self, ax: Axes3D) -> None:
        """Per-frame 3D axes setup: layering, theme, and the rich court."""
        enable_explicit_layering(ax)
        apply_axes_theme_3d(ax, self.theme)
        x_half_span = float(HALF_DOUBLES_WIDTH + _VIEW_MARGIN)
        y_half_span = float(HALF_LENGTH + _VIEW_MARGIN)
        self.court_renderer.render_3d(
            ax,
            show_net=True,
            apron_bounds=(-x_half_span, x_half_span, -y_half_span, y_half_span),
        )

    def _render_ball_3d_frame(
        self,
        ax: Axes3D,
        positions: NDArray[np.float32],
        frame_idx: int,
        *,
        color: str | None = None,
        label: str | None = None,
    ) -> None:
        """Draw one trajectory's fading trail, shadow, and current ball."""
        trail_color = color or self.ball_renderer.style.trajectory_color
        if self.style.show_trail:
            start_idx = max(0, frame_idx - self.style.trail_length)
            render_fading_line_3d(
                ax,
                positions[start_idx : frame_idx + 1],
                color=trail_color,
                alpha_range=(0.05, 0.95),
                linewidth_range=(1.0, 3.0),
                zorder=SceneLayer.TRAIL,
            )

        ball_pos = positions[frame_idx]
        if not np.isfinite(ball_pos).all():
            return
        if self.style.show_shadow:
            # Fade the contact shadow out as the ball rises.
            height_ratio = float(np.clip(ball_pos[2] / _VIEW_Z_LIMIT, 0.0, 1.0))
            render_ground_shadow(
                ax,
                (float(ball_pos[0]), float(ball_pos[1])),
                radius=_BALL_SHADOW_RADIUS,
                alpha=0.35 * (1.0 - 0.7 * height_ratio),
                zorder=SceneLayer.GROUND,
            )
        style_override = BallStyle(ball_color=color) if color is not None else None
        self.ball_renderer.render_ball_3d(
            ax,
            ball_pos,
            label=label,
            style_override=style_override,
            zorder=SceneLayer.BALL,
        )

    def _render_bounce_rings(
        self,
        ax: Axes3D,
        positions: NDArray[np.float32],
        bounce_frames: NDArray[np.int64],
        frame_idx: int,
        fps: float,
    ) -> None:
        duration_frames = max(1, int(round(_BOUNCE_MARKER_DURATION_S * fps)))
        for b in bounce_frames.tolist():
            age_frames = frame_idx - b
            if age_frames < 0 or age_frames > duration_frames:
                continue
            pos = positions[b]
            if not np.isfinite(pos[:2]).all():
                continue
            render_impact_ring(
                ax,
                (float(pos[0]), float(pos[1])),
                age_frames / duration_frames,
                color=_BOUNCE_RING_COLOR,
                zorder=SceneLayer.RING,
            )

    def _render_minimap_frame(
        self,
        minimap_ax: Axes,
        trajectories: list[tuple[NDArray[np.float32], str]],
        bounce_positions_xy: NDArray[np.float32] | None,
        frame_idx: int,
    ) -> None:
        trails = []
        trail_dots = []
        for positions, color in trajectories:
            trail_start = max(0, frame_idx - _MINIMAP_TRAIL_FRAMES)
            trails.append((positions[trail_start : frame_idx + 1, :2], color))
            pos = positions[frame_idx]
            trail_dots.append(((float(pos[0]), float(pos[1])), color))
        self.minimap_renderer.render(
            minimap_ax,
            trails=trails,
            trail_dots=trail_dots,
            event_marks_xy=bounce_positions_xy,
        )

    def create_animation(
        self,
        scene: dict[str, Any],
        view: str = "2d",
        camera_idx: int = 0,
        *,
        fps: float = 30.0,
        figsize: tuple[float, float] = (10, 8),
    ) -> FuncAnimation | None:
        """Create animation of ball trajectory.

        Args:
            scene: BLCS scene dictionary.
            view: View type ('3d', '2d', 'camera').
            camera_idx: Camera index for camera view.
            fps: Frames per second.
            figsize: Figure size.

        Returns:
            FuncAnimation object, or None if view is invalid.

        """
        tracks = split_ball_tracks(scene)
        num_frames = len(tracks[0])
        interval = 1000.0 / fps

        if view == "3d":
            bounce_frames = [
                resolve_bounce_frames(
                    positions,
                    extract_ball_track_events(scene["meta"], index),
                )
                for index, positions in enumerate(tracks)
            ]
            speeds = (
                [compute_speeds(positions, fps) for positions in tracks]
                if self.style.show_hud
                else None
            )

            fig = plt.figure(figsize=figsize)
            apply_figure_theme(fig, self.theme)
            ax = fig.add_subplot(111, projection="3d")
            apply_axes_layout_3d(ax, self.theme)
            minimap_ax = (
                fig.add_axes(_MINIMAP_RECT) if self.style.show_minimap else None
            )

            def update_3d(frame_idx: int) -> list:
                ax.clear()
                self._setup_3d_axes(ax)
                for index, positions in enumerate(tracks):
                    color = _BALL_COLORS[index % len(_BALL_COLORS)]
                    self._render_bounce_rings(
                        ax, positions, bounce_frames[index], frame_idx, fps
                    )
                    self._render_ball_3d_frame(
                        ax,
                        positions,
                        frame_idx,
                        color=color,
                        label=f"Ball {index + 1}",
                    )
                if len(tracks) > 1:
                    ax.legend(loc="upper right")
                if self.style.show_hud:
                    assert speeds is not None
                    lines = [
                        format_frame_clock(frame_idx, num_frames - 1, fps),
                        *[
                            f"Ball {index + 1} speed "
                            f"{format_speed_kmh(float(track_speeds[frame_idx]))}"
                            for index, track_speeds in enumerate(speeds)
                        ],
                        "Bounces "
                        f"{sum(int((frames <= frame_idx).sum()) for frames in bounce_frames)}",
                    ]
                    render_hud_text(ax, lines, self.hud_style)
                view_now = self.camera.view_at(frame_idx, fps)
                apply_scene_camera(
                    ax, view_now, margin=_VIEW_MARGIN, z_limit=_VIEW_Z_LIMIT
                )
                if self.theme.name != "dark":
                    ax.set_title(
                        f"Frame {frame_idx}/{num_frames - 1}",
                        color=self.theme.text_color,
                    )
                if minimap_ax is not None:
                    minimap_ax.clear()
                    bounce_positions = [
                        positions[frames[frames <= frame_idx], :2]
                        for positions, frames in zip(tracks, bounce_frames, strict=True)
                    ]
                    nonempty_bounces = [
                        value for value in bounce_positions if len(value)
                    ]
                    self._render_minimap_frame(
                        minimap_ax,
                        [
                            (positions, _BALL_COLORS[index % len(_BALL_COLORS)])
                            for index, positions in enumerate(tracks)
                        ],
                        np.concatenate(nonempty_bounces) if nonempty_bounces else None,
                        frame_idx,
                    )
                return []

            return FuncAnimation(
                fig, update_3d, frames=num_frames, interval=interval, blit=False
            )

        elif view == "2d":
            fig, ax = plt.subplots(figsize=figsize)
            self.court_renderer.render_2d(ax, show_fence=True, set_limits=False)

            lines = [
                ax.plot([], [], color=color, linewidth=2)[0]
                for color in _BALL_COLORS[: len(tracks)]
            ]
            points = [
                ax.scatter([], [], c=color, s=100, zorder=10)
                for color in _BALL_COLORS[: len(tracks)]
            ]

            ax.set_xlim(-HALF_DOUBLES_WIDTH - 2, HALF_DOUBLES_WIDTH + 2)
            ax.set_ylim(-HALF_LENGTH - 2, HALF_LENGTH + 2)
            ax.set_aspect("equal")

            def update_2d(frame: int) -> tuple:
                for positions, line, point in zip(tracks, lines, points, strict=True):
                    line.set_data(positions[: frame + 1, 0], positions[: frame + 1, 1])
                    point.set_offsets([[positions[frame, 0], positions[frame, 1]]])
                ax.set_title(f"Top-down | Frame {frame}/{num_frames - 1}")
                return (*lines, *points)

            return FuncAnimation(
                fig, update_2d, frames=num_frames, interval=interval, blit=False
            )

        elif view == "camera":
            if camera_idx >= scene["num_cameras"]:
                print(f"Error: Camera {camera_idx} out of range")
                return None

            cam = scene["cameras"][camera_idx]
            ball_uv = np.asarray(cam["ball_uv"])
            uv_tracks = (
                [ball_uv]
                if ball_uv.ndim == 2
                else [ball_uv[:, index] for index in range(len(tracks))]
            )

            fig, ax = plt.subplots(figsize=figsize)
            court_uv = cam["court_kp_uv"]
            court_vis = cam["court_kp_visible"]
            self.court_renderer.render_projected_2d(
                ax,
                court_uv,
                court_vis,
                line_color="lime",
                line_width=1.5,
                visible_line_alpha=0.8,
                keypoint_color="lime",
                keypoint_size=30.0,
                keypoint_alpha=0.7,
                keypoint_marker="s",
            )

            lines = [
                ax.plot([], [], color=color, linewidth=1, alpha=0.5)[0]
                for color in _BALL_COLORS[: len(uv_tracks)]
            ]
            points = [
                ax.scatter([], [], c=color, s=100, zorder=10)
                for color in _BALL_COLORS[: len(uv_tracks)]
            ]

            ax.set_xlim(0, 1)
            ax.set_ylim(1, 0)
            ax.grid(True, alpha=0.3)

            def update_cam(frame: int) -> tuple:
                for uv_track, line, point in zip(uv_tracks, lines, points, strict=True):
                    line.set_data(uv_track[: frame + 1, 0], uv_track[: frame + 1, 1])
                    point.set_offsets([[uv_track[frame, 0], uv_track[frame, 1]]])
                ax.set_title(f"Camera {camera_idx} | Frame {frame}/{num_frames - 1}")
                return (*lines, *points)

            return FuncAnimation(
                fig, update_cam, frames=num_frames, interval=interval, blit=False
            )

        else:
            print(f"Unknown view type: {view}. Use '3d', '2d', or 'camera'.")
            return None

    def create_comparison_animation(
        self,
        gt_positions: Any,
        pred_positions: Any,
        view: str = "3d",
        *,
        fps: float = 30.0,
        figsize: tuple[float, float] = (10, 8),
        title: str = "GT vs Prediction",
        events: list[BallEvent] | None = None,
    ) -> FuncAnimation | None:
        """Create animation comparing GT and predicted trajectories.

        Args:
            gt_positions: Ground truth ball positions (T, 3).
            pred_positions: Predicted ball positions (T, 3).
            view: View type ('3d' or '2d').
            fps: Frames per second.
            figsize: Figure size.
            title: Title prefix for the animation.
            events: GT ball events from scene metadata; bounce rings prefer
                these over trajectory-based detection (3D view only).

        Returns:
            FuncAnimation object, or None if view is invalid.

        """
        gt_positions = np.asarray(gt_positions)
        pred_positions = np.asarray(pred_positions)
        num_frames = len(gt_positions)
        interval = 1000.0 / fps

        if view == "3d":
            # Bounce rings mark GT bounces only; a second ring set from the
            # prediction would double-mark the same physical bounce.
            bounce_frames = resolve_bounce_frames(gt_positions, events)

            fig = plt.figure(figsize=figsize)
            apply_figure_theme(fig, self.theme)
            ax = fig.add_subplot(111, projection="3d")
            apply_axes_layout_3d(ax, self.theme)
            minimap_ax = (
                fig.add_axes(_MINIMAP_RECT) if self.style.show_minimap else None
            )

            def update_3d(frame_idx: int) -> list:
                ax.clear()
                self._setup_3d_axes(ax)
                self._render_bounce_rings(
                    ax, gt_positions, bounce_frames, frame_idx, fps
                )
                self._render_ball_3d_frame(
                    ax, gt_positions, frame_idx, color=_GT_COLOR, label="GT"
                )
                self._render_ball_3d_frame(
                    ax, pred_positions, frame_idx, color=_PRED_COLOR, label="Prediction"
                )
                ax.legend(loc="upper right")
                if self.style.show_hud:
                    render_hud_text(
                        ax,
                        [format_frame_clock(frame_idx, num_frames - 1, fps)],
                        self.hud_style,
                    )
                view_now = self.camera.view_at(frame_idx, fps)
                apply_scene_camera(
                    ax, view_now, margin=_VIEW_MARGIN, z_limit=_VIEW_Z_LIMIT
                )
                if self.theme.name != "dark":
                    ax.set_title(
                        f"{title} | Frame {frame_idx}/{num_frames - 1}",
                        color=self.theme.text_color,
                    )
                if minimap_ax is not None:
                    minimap_ax.clear()
                    past = bounce_frames[bounce_frames <= frame_idx]
                    self._render_minimap_frame(
                        minimap_ax,
                        [(gt_positions, _GT_COLOR), (pred_positions, _PRED_COLOR)],
                        gt_positions[past, :2],
                        frame_idx,
                    )
                return []

            return FuncAnimation(
                fig, update_3d, frames=num_frames, interval=interval, blit=False
            )

        elif view == "2d":
            fig, ax = plt.subplots(figsize=figsize)
            self.court_renderer.render_2d(ax, show_fence=True, set_limits=False)

            # GT trajectory (green)
            (gt_line,) = ax.plot([], [], "g-", linewidth=2, label="GT")
            gt_point = ax.scatter([], [], c="green", s=100, zorder=10, marker="o")

            # Predicted trajectory (red)
            (pred_line,) = ax.plot([], [], "r-", linewidth=2, label="Prediction")
            pred_point = ax.scatter([], [], c="red", s=100, zorder=10, marker="^")

            ax.set_xlim(-HALF_DOUBLES_WIDTH - 2, HALF_DOUBLES_WIDTH + 2)
            ax.set_ylim(-HALF_LENGTH - 2, HALF_LENGTH + 2)
            ax.set_aspect("equal")
            ax.legend(loc="upper right")

            def update_2d(frame: int) -> tuple:
                # GT
                gt_line.set_data(
                    gt_positions[: frame + 1, 0], gt_positions[: frame + 1, 1]
                )
                gt_point.set_offsets([[gt_positions[frame, 0], gt_positions[frame, 1]]])
                # Prediction
                pred_line.set_data(
                    pred_positions[: frame + 1, 0], pred_positions[: frame + 1, 1]
                )
                pred_point.set_offsets(
                    [[pred_positions[frame, 0], pred_positions[frame, 1]]]
                )
                ax.set_title(f"{title} | Frame {frame}/{num_frames - 1}")
                return gt_line, gt_point, pred_line, pred_point

            return FuncAnimation(
                fig, update_2d, frames=num_frames, interval=interval, blit=False
            )

        else:
            print(f"Unknown view type for comparison: {view}. Use '3d' or '2d'.")
            return None

    def print_scene_info(self, scene: dict[str, Any]) -> None:
        """Print scene metadata and statistics.

        Args:
            scene: BLCS scene dictionary.

        """
        meta = scene["meta"]
        print("=" * 60)
        print("SCENE INFORMATION")
        print("=" * 60)
        print(f"Scene ID: {meta.get('scene_id', 'Unknown')}")

        print("Type: Rally Scene")
        print(f"Rally Length: {meta.get('rally_length', 'N/A')} shots")
        print(f"End Reason: {meta.get('end_reason', 'N/A')}")
        print(f"Winner: {meta.get('winner_side', 'N/A')}")
        print(
            f"Initial From: Cell {meta.get('initial_from_cell', 'N/A')}, "
            f"Side {meta.get('initial_from_side', 'N/A')}"
        )

        shots = meta.get("shots", [])
        print(f"\nShot Breakdown ({len(shots)} shots):")
        for shot in shots:
            print(f"  Shot {shot.get('shot_index', '?') + 1}:")
            print(
                f"    From: Cell {shot.get('from_cell', '?')}, "
                f"Side {shot.get('from_side', '?')}"
            )
            print(f"    Category: {shot.get('category', '?')}")
            print(f"    To Cell: {shot.get('to_cell', '?')}")
            print(
                f"    Events: t_start={shot.get('t_start', -1)}, "
                f"t_net={shot.get('t_net', -1)}, "
                f"t_bounce1={shot.get('t_bounce1', -1)}, "
                f"t_return={shot.get('t_return', -1)}"
            )

        # Common info
        print("\nTrajectory:")
        print(f"  Frames: {meta.get('num_frames', 'N/A')}")
        print(
            f"  FPS: {meta.get('fps_out', 'N/A')} (output), "
            f"{meta.get('sim_fps', 'N/A')} (sim)"
        )
        fps_out = meta.get("fps_out")
        num_frames = meta.get("num_frames")
        if fps_out and num_frames:
            print(f"  Duration: {num_frames / fps_out:.2f} seconds")

        # Position statistics
        pos = scene["ball_pos_world"]
        print("\nBall position statistics (world coordinates, meters):")
        print(f"  X range: [{pos[:, 0].min():.2f}, {pos[:, 0].max():.2f}]")
        print(f"  Y range: [{pos[:, 1].min():.2f}, {pos[:, 1].max():.2f}]")
        print(f"  Z range: [{pos[:, 2].min():.2f}, {pos[:, 2].max():.2f}]")

        # Camera info
        num_cameras = scene["num_cameras"]
        print(
            f"\nCameras: {num_cameras} valid "
            f"(from {meta.get('num_cameras_sampled', 'N/A')} sampled)"
        )
        print("Camera visibility:")
        for i, cam in enumerate(scene["cameras"]):
            print(
                f"  Camera {i}: Ball {cam['ball_visibility_ratio']:.1%}, "
                f"Court {cam['court_visibility_count']:.1f}/20"
            )
        print("=" * 60)
