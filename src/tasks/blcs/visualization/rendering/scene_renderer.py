"""BLCS scene renderer for ball trajectory visualization.

This module provides complete scene rendering for BLCS (Ball Location from
Court keypoints and Skeleton) data, combining court and ball trajectory
visualization.

Example:
    >>> from src.tasks.blcs.visualization.rendering import BLCSSceneRenderer
    >>>
    >>> renderer = BLCSSceneRenderer()
    >>> fig = renderer.render_multi_view(scene_data, frame_idx=0)
    >>> plt.show()

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.utils.schema.court import HALF_DOUBLES_WIDTH, HALF_LENGTH
from src.utils.rendering.ball_renderer import (
    BallEvent,
    BallEventType,
    BallRenderer,
    BallStyle,
)
from src.utils.rendering.court_renderer import CourtRenderer

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D


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
        court_renderer: CourtRenderer | None = None,
        ball_renderer: BallRenderer | None = None,
    ) -> None:
        """Initialize BLCS scene renderer.

        Args:
            court_renderer: Court renderer instance. If None, creates default.
            ball_renderer: Ball renderer. If None, creates default.

        """
        self.court_renderer = court_renderer or CourtRenderer()
        self.ball_renderer = ball_renderer or BallRenderer()

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
        """Render 3D view of ball trajectory.

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
            ax = fig.add_subplot(111, projection="3d")

        # Render court
        self.court_renderer.render_3d(ax, show_net=True)

        # Get ball trajectory
        positions = scene["ball_pos_world"]
        meta = scene["meta"]

        # Build event list
        events = self._extract_events(meta)

        # Render trajectory
        highlight = frame_idx if frame_idx >= 0 else None
        self.ball_renderer.render_trajectory_3d(
            ax,
            positions,
            events=events,
            highlight_frame=highlight,
        )

        # Title
        ax.set_title(self._get_display_title(meta))

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

        # Get ball trajectory
        positions = scene["ball_pos_world"]
        meta = scene["meta"]

        # Build event list
        events = self._extract_events(meta)

        # Create style with height colormap option
        style = BallStyle(use_height_colormap=use_height_colormap)

        # Render trajectory
        highlight = frame_idx if frame_idx >= 0 else None
        self.ball_renderer.render_trajectory_2d(
            ax,
            positions,
            events=events,
            highlight_frame=highlight,
            style_override=style,
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

        # Draw court keypoints
        for i in range(20):
            if court_vis[i]:
                ax.scatter(
                    court_uv[i, 0],
                    court_uv[i, 1],
                    c="lime",
                    s=50,
                    marker="s",
                    alpha=0.7,
                )

        # Build events
        events = self._extract_events(meta)

        # Render ball trajectory in UV space
        self.ball_renderer.render_trajectory_uv(
            ax,
            ball_uv,
            visibility=ball_vis.astype(bool),
            events=events,
        )

        # Highlight specific frame
        if frame_idx >= 0 and frame_idx < len(ball_uv):
            ax.scatter(
                [ball_uv[frame_idx, 0]],
                [ball_uv[frame_idx, 1]],
                c="blue",
                s=200,
                marker="o",
                edgecolors="white",
                linewidths=3,
                zorder=15,
                label=f"Frame {frame_idx}",
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
        fig = plt.figure(figsize=figsize)

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
        positions = scene["ball_pos_world"]
        num_frames = len(positions)
        meta = scene["meta"]
        events = self._extract_events(meta)
        interval = 1000.0 / fps

        if view == "3d":
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
            self.court_renderer.render_3d(ax, show_net=True)

            (line,) = ax.plot([], [], [], "r-", linewidth=2)
            point = ax.scatter([], [], [], c="red", s=100)

            ax.set_xlim(-HALF_DOUBLES_WIDTH - 2, HALF_DOUBLES_WIDTH + 2)
            ax.set_ylim(-HALF_LENGTH - 2, HALF_LENGTH + 2)
            ax.set_zlim(0, 5)

            def update_3d(frame: int) -> tuple:
                line.set_data(positions[: frame + 1, 0], positions[: frame + 1, 1])
                line.set_3d_properties(positions[: frame + 1, 2])
                point._offsets3d = (
                    [positions[frame, 0]],
                    [positions[frame, 1]],
                    [positions[frame, 2]],
                )
                ax.set_title(f"Frame {frame}/{num_frames - 1}")
                return line, point

            return FuncAnimation(
                fig, update_3d, frames=num_frames, interval=interval, blit=False
            )

        elif view == "2d":
            fig, ax = plt.subplots(figsize=figsize)
            self.court_renderer.render_2d(ax, show_fence=True, set_limits=False)

            (line,) = ax.plot([], [], "r-", linewidth=2)
            point = ax.scatter([], [], c="red", s=100, zorder=10)

            ax.set_xlim(-HALF_DOUBLES_WIDTH - 2, HALF_DOUBLES_WIDTH + 2)
            ax.set_ylim(-HALF_LENGTH - 2, HALF_LENGTH + 2)
            ax.set_aspect("equal")

            def update_2d(frame: int) -> tuple:
                line.set_data(positions[: frame + 1, 0], positions[: frame + 1, 1])
                point.set_offsets([[positions[frame, 0], positions[frame, 1]]])
                ax.set_title(f"Top-down | Frame {frame}/{num_frames - 1}")
                return line, point

            return FuncAnimation(
                fig, update_2d, frames=num_frames, interval=interval, blit=False
            )

        elif view == "camera":
            if camera_idx >= scene["num_cameras"]:
                print(f"Error: Camera {camera_idx} out of range")
                return None

            cam = scene["cameras"][camera_idx]
            ball_uv = cam["ball_uv"]

            fig, ax = plt.subplots(figsize=figsize)

            # Draw court lines (static)
            from src.utils.schema.court import COURT_SKELETON

            court_uv = cam["court_kp_uv"]
            court_vis = cam["court_kp_visible"]

            for i, j in COURT_SKELETON:
                if court_vis[i] and court_vis[j]:
                    ax.plot(
                        [court_uv[i, 0], court_uv[j, 0]],
                        [court_uv[i, 1], court_uv[j, 1]],
                        c="lime",
                        linewidth=1.5,
                        alpha=0.8,
                    )

            # Draw court keypoints (static)
            for i in range(20):
                if court_vis[i]:
                    ax.scatter(
                        court_uv[i, 0],
                        court_uv[i, 1],
                        c="lime",
                        s=30,
                        marker="s",
                        alpha=0.7,
                    )

            (line,) = ax.plot([], [], "r-", linewidth=1, alpha=0.5)
            point = ax.scatter([], [], c="red", s=100, zorder=10)

            ax.set_xlim(0, 1)
            ax.set_ylim(1, 0)
            ax.grid(True, alpha=0.3)

            def update_cam(frame: int) -> tuple:
                line.set_data(ball_uv[: frame + 1, 0], ball_uv[: frame + 1, 1])
                point.set_offsets([[ball_uv[frame, 0], ball_uv[frame, 1]]])
                ax.set_title(f"Camera {camera_idx} | Frame {frame}/{num_frames - 1}")
                return line, point

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
    ) -> FuncAnimation | None:
        """Create animation comparing GT and predicted trajectories.

        Args:
            gt_positions: Ground truth ball positions (T, 3).
            pred_positions: Predicted ball positions (T, 3).
            view: View type ('3d' or '2d').
            fps: Frames per second.
            figsize: Figure size.
            title: Title prefix for the animation.

        Returns:
            FuncAnimation object, or None if view is invalid.

        """
        import numpy as np

        gt_positions = np.asarray(gt_positions)
        pred_positions = np.asarray(pred_positions)
        num_frames = len(gt_positions)
        interval = 1000.0 / fps

        if view == "3d":
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
            self.court_renderer.render_3d(ax, show_net=True)

            # GT trajectory (green)
            (gt_line,) = ax.plot([], [], [], "g-", linewidth=2, label="GT")
            gt_point = ax.scatter([], [], [], c="green", s=100, marker="o")

            # Predicted trajectory (red)
            (pred_line,) = ax.plot([], [], [], "r-", linewidth=2, label="Prediction")
            pred_point = ax.scatter([], [], [], c="red", s=100, marker="^")

            ax.set_xlim(-HALF_DOUBLES_WIDTH - 2, HALF_DOUBLES_WIDTH + 2)
            ax.set_ylim(-HALF_LENGTH - 2, HALF_LENGTH + 2)
            ax.set_zlim(0, 5)
            ax.legend(loc="upper right")

            def update_3d(frame: int) -> tuple:
                # GT
                gt_line.set_data(gt_positions[: frame + 1, 0], gt_positions[: frame + 1, 1])
                gt_line.set_3d_properties(gt_positions[: frame + 1, 2])
                gt_point._offsets3d = (
                    [gt_positions[frame, 0]],
                    [gt_positions[frame, 1]],
                    [gt_positions[frame, 2]],
                )
                # Prediction
                pred_line.set_data(pred_positions[: frame + 1, 0], pred_positions[: frame + 1, 1])
                pred_line.set_3d_properties(pred_positions[: frame + 1, 2])
                pred_point._offsets3d = (
                    [pred_positions[frame, 0]],
                    [pred_positions[frame, 1]],
                    [pred_positions[frame, 2]],
                )
                ax.set_title(f"{title} | Frame {frame}/{num_frames - 1}")
                return gt_line, gt_point, pred_line, pred_point

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
                gt_line.set_data(gt_positions[: frame + 1, 0], gt_positions[: frame + 1, 1])
                gt_point.set_offsets([[gt_positions[frame, 0], gt_positions[frame, 1]]])
                # Prediction
                pred_line.set_data(pred_positions[: frame + 1, 0], pred_positions[: frame + 1, 1])
                pred_point.set_offsets([[pred_positions[frame, 0], pred_positions[frame, 1]]])
                ax.set_title(f"{title} | Frame {frame}/{num_frames - 1}")
                return gt_line, gt_point, pred_line, pred_point

            return FuncAnimation(
                fig, update_2d, frames=num_frames, interval=interval, blit=False
            )

        else:
            print(f"Unknown view type for comparison: {view}. Use '3d' or '2d'.")
            return None

    def _extract_events(self, meta: dict[str, Any]) -> list[BallEvent]:
        """Extract ball events from scene metadata.

        Args:
            meta: Scene metadata dictionary.

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
        print(f"\nTrajectory:")
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
