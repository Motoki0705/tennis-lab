"""PLCS scene renderer for player pose visualization.

This module provides complete scene rendering for PLCS (Player Location
and pose from Court keypoints and Skeleton) data, combining court and
skeleton visualization.

Example:
    >>> from src.plcs.visualization.rendering import PLCSSceneRenderer
    >>>
    >>> renderer = PLCSSceneRenderer()
    >>> fig, ax = renderer.render_frame_3d(scene_data, frame_idx=0)
    >>> plt.show()

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from src.utils.schema.keypoint_schema import COURT_LINE_CONNECTIONS
from src.utils.geometry.court import HALF_DOUBLES_WIDTH, HALF_LENGTH, NET_HEIGHT_POST
from src.utils.rendering.court_renderer import CourtRenderer
from src.utils.rendering.skeleton_renderer import SkeletonRenderer

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D


# Court keypoint skeleton for drawing lines between keypoints
# Use the unified definition from schema.keypoint_schema
COURT_SKELETON: list[tuple[int, int]] = COURT_LINE_CONNECTIONS


class PLCSSceneRenderer:
    """Render complete PLCS scenes with court and player skeletons.

    Combines court rendering and skeleton rendering for comprehensive
    scene visualization in 2D and 3D views.

    Example:
        >>> renderer = PLCSSceneRenderer()
        >>> fig, ax = renderer.render_frame_3d(scene, frame_idx=0)
        >>> anim = renderer.create_animation(scene, view="2d_topdown")

    """

    def __init__(
        self,
        court_renderer: CourtRenderer | None = None,
        skeleton_renderer: SkeletonRenderer | None = None,
    ) -> None:
        """Initialize PLCS scene renderer.

        Args:
            court_renderer: Court renderer instance. If None, creates default.
            skeleton_renderer: Skeleton renderer. If None, creates COCO-17 renderer.

        """
        self.court_renderer = court_renderer or CourtRenderer()
        self.skeleton_renderer = skeleton_renderer or SkeletonRenderer(
            skeleton_type="coco17"
        )
        # Also keep a SMPL-H renderer for canonical poses
        self.smplh_renderer = SkeletonRenderer(skeleton_type="smplh")

    def render_frame_3d(
        self,
        scene: Any,
        frame_idx: int,
        *,
        ax: Axes3D | None = None,
        clear_axes: bool = True,
        figsize: tuple[float, float] = (12, 8),
        show_direction: bool = True,
        title: str | None = None,
    ) -> tuple[Figure, Axes3D]:
        """Render a single frame in 3D.

        Args:
            scene: PLCS scene data object with position, rotation, canonical_pose_3d.
            frame_idx: Frame index to render.
            ax: Optional existing 3D axes. If None, creates new figure.
            clear_axes: Whether to clear axes before rendering (for overlay).
            figsize: Figure size in inches.
            show_direction: Whether to show player facing direction arrow.
            title: Custom title. If None, generates from scene metadata.

        Returns:
            Tuple of (figure, 3D axes).

        """
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.get_figure()
            if clear_axes:
                ax.clear()

        # Render court
        self.court_renderer.render_3d(ax, show_net=True)

        # Get player position (denormalized from [-1, 1] to meters)
        pos = scene.position[frame_idx]
        x = pos[0] * HALF_DOUBLES_WIDTH
        y = pos[1] * HALF_LENGTH
        z = pos[2] * NET_HEIGHT_POST if len(pos) > 2 else 0

        # Get rotation (sin, cos representation)
        sin_yaw = scene.rotation[frame_idx, 0]
        cos_yaw = scene.rotation[frame_idx, 1]
        yaw = np.arctan2(sin_yaw, cos_yaw)

        # Rotation matrix for transforming canonical pose
        rot = np.array(
            [
                [cos_yaw, -sin_yaw, 0],
                [sin_yaw, cos_yaw, 0],
                [0, 0, 1],
            ]
        )

        # Transform canonical pose to world coordinates
        canonical_pose = scene.canonical_pose_3d[frame_idx]  # (J, 3)
        world_pose = canonical_pose @ rot.T
        world_pose[:, 0] += x
        world_pose[:, 1] += y
        world_pose[:, 2] += z

        # Render skeleton
        self.smplh_renderer.render_3d(ax, world_pose)

        # Draw direction arrow
        if show_direction:
            arrow_length = 1.0
            dx = arrow_length * (-sin_yaw)
            dy = arrow_length * cos_yaw
            ax.quiver(
                x,
                y,
                z + 1.5,
                dx,
                dy,
                0,
                color="red",
                arrow_length_ratio=0.3,
                linewidth=2,
            )

        # Set title
        if title is None:
            meta = getattr(scene, "meta", {})
            scene_id = meta.get("scene_id", "unknown")
            num_frames = meta.get("num_frames", "?")
            category = meta.get("motion_category", "unknown")
            title = f"Scene: {scene_id} | Frame: {frame_idx}/{num_frames} | {category}"
        ax.set_title(title)

        return fig, ax

    def render_frame_2d_topdown(
        self,
        scene: Any,
        frame_idx: int,
        *,
        ax: Axes | None = None,
        clear_axes: bool = True,
        figsize: tuple[float, float] = (10, 12),
        show_direction: bool = True,
        show_trail: bool = True,
        trail_length: int = 30,
        title: str | None = None,
    ) -> tuple[Figure, Axes]:
        """Render a single frame as 2D top-down view.

        Args:
            scene: PLCS scene data object.
            frame_idx: Frame index to render.
            ax: Optional existing axes. If None, creates new figure.
            clear_axes: Whether to clear axes before rendering (for overlay).
            figsize: Figure size in inches.
            show_direction: Whether to show facing direction arrow.
            show_trail: Whether to show movement trail.
            trail_length: Number of past frames for trail.
            title: Custom title.

        Returns:
            Tuple of (figure, axes).

        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
            if clear_axes:
                ax.clear()

        # Render court
        self.court_renderer.render_2d(ax, show_fence=True)

        # Get player position
        pos = scene.position[frame_idx]
        x = pos[0] * HALF_DOUBLES_WIDTH
        y = pos[1] * HALF_LENGTH

        # Draw movement trail
        if show_trail:
            start_idx = max(0, frame_idx - trail_length)
            trail_pos = scene.position[start_idx : frame_idx + 1]
            trail_x = trail_pos[:, 0] * HALF_DOUBLES_WIDTH
            trail_y = trail_pos[:, 1] * HALF_LENGTH
            ax.plot(trail_x, trail_y, "c-", linewidth=2, alpha=0.5, zorder=4)

        # Draw player position
        ax.scatter([x], [y], c="red", s=100, zorder=5, label="Player")

        # Draw direction arrow
        if show_direction:
            sin_yaw = scene.rotation[frame_idx, 0]
            cos_yaw = scene.rotation[frame_idx, 1]
            arrow_length = 1.5
            dx = arrow_length * (-sin_yaw)
            dy = arrow_length * cos_yaw
            ax.arrow(
                x,
                y,
                dx,
                dy,
                head_width=0.3,
                head_length=0.2,
                fc="yellow",
                ec="black",
                zorder=6,
            )

        # Set title
        if title is None:
            meta = getattr(scene, "meta", {})
            scene_id = meta.get("scene_id", "unknown")
            num_frames = meta.get("num_frames", "?")
            category = meta.get("motion_category", "unknown")
            title = f"Scene: {scene_id} | Frame: {frame_idx}/{num_frames} | {category}"
        ax.set_title(title)

        return fig, ax

    def render_camera_view(
        self,
        scene: Any,
        frame_idx: int,
        camera_idx: int,
        *,
        figsize: tuple[float, float] = (12, 8),
        show_court_lines: bool = True,
        court_kp_size: float = 30.0,
    ) -> tuple[Figure, Axes]:
        """Render a camera view (2D UV space).

        Args:
            scene: PLCS scene data object.
            frame_idx: Frame index to render.
            camera_idx: Camera index.
            figsize: Figure size.
            show_court_lines: Whether to draw court line connections.
            court_kp_size: Size of court keypoint markers.

        Returns:
            Tuple of (figure, axes).

        Raises:
            ValueError: If camera_idx is out of range.

        """
        if camera_idx >= len(scene.cameras):
            raise ValueError(
                f"Camera index {camera_idx} out of range (max: {len(scene.cameras) - 1})"
            )

        cam = scene.cameras[camera_idx]

        fig, ax = plt.subplots(figsize=figsize)

        # Set up UV coordinate space
        ax.set_facecolor("#1a1a1a")
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)  # Flip Y for image coordinates

        # Draw court keypoints
        court_uv = cam.court_kp_uv[frame_idx]
        court_vis = cam.court_kp_visible[frame_idx]
        visible_court = court_uv[court_vis]
        ax.scatter(
            visible_court[:, 0],
            visible_court[:, 1],
            c="lime",
            s=court_kp_size,
            marker="s",
            label=f"Court KP ({court_vis.sum()}/20)",
        )

        # Draw court lines
        if show_court_lines:
            for i, j in COURT_SKELETON:
                alpha = 0.5 if (court_vis[i] and court_vis[j]) else 0.2
                ax.plot(
                    [court_uv[i, 0], court_uv[j, 0]],
                    [court_uv[i, 1], court_uv[j, 1]],
                    color="lime",
                    linewidth=1,
                    alpha=alpha,
                )

        # Draw human keypoints
        human_uv = cam.human_kp_uv[frame_idx]
        human_vis = cam.human_kp_visible[frame_idx]
        self.skeleton_renderer.render_2d(ax, human_uv, human_vis, label="Human")

        # Title with visibility stats
        ax.set_title(
            f"Camera {camera_idx} | Frame: {frame_idx} | "
            f"Human vis: {cam.human_visibility_ratio:.1%} | "
            f"Court vis: {cam.court_visibility_count:.0f}/20"
        )
        ax.legend(loc="upper right")
        ax.set_xlabel("U (normalized)")
        ax.set_ylabel("V (normalized)")

        return fig, ax

    def render_multi_view(
        self,
        scene: Any,
        frame_idx: int,
        *,
        figsize: tuple[float, float] = (16, 12),
    ) -> tuple[Figure, list[Axes]]:
        """Render multiple views in a grid layout.

        Shows 3D view, 2D top-down, and available camera views.

        Args:
            scene: PLCS scene data object.
            frame_idx: Frame index.
            figsize: Figure size.

        Returns:
            Tuple of (figure, list of axes).

        """
        num_cameras = len(scene.cameras)
        cols = min(3, num_cameras + 2)
        rows = (num_cameras + 2 + cols - 1) // cols

        fig = plt.figure(figsize=figsize)

        # 3D view
        ax_3d = fig.add_subplot(rows, cols, 1, projection="3d")
        self._render_3d_subplot(ax_3d, scene, frame_idx)
        ax_3d.set_title("3D View")

        # 2D top-down view
        ax_2d = fig.add_subplot(rows, cols, 2)
        self._render_2d_subplot(ax_2d, scene, frame_idx)
        ax_2d.set_title("Top-Down View")

        # Camera views
        for i in range(num_cameras):
            ax = fig.add_subplot(rows, cols, 3 + i)
            self._render_camera_subplot(ax, scene, frame_idx, i)

        plt.tight_layout()
        return fig, fig.axes

    def create_animation(
        self,
        scene: Any,
        view: str = "3d",
        camera_idx: int = 0,
        *,
        fps: float | None = None,
        figsize: tuple[float, float] = (12, 8),
    ) -> FuncAnimation:
        """Create animation of scene.

        Args:
            scene: PLCS scene data object.
            view: View type ('3d', '2d_topdown', 'camera').
            camera_idx: Camera index for camera view.
            fps: Frames per second. If None, uses scene FPS.
            figsize: Figure size.

        Returns:
            FuncAnimation object.

        Raises:
            ValueError: If view type is unknown.

        """
        meta = getattr(scene, "meta", {})
        if fps is None:
            fps = meta.get("fps", 30.0)

        num_frames = meta.get("num_frames", len(scene.position))
        interval = 1000.0 / fps

        if view == "3d":
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")

            def update_3d(frame_idx: int) -> list:
                ax.clear()
                self._render_3d_subplot(ax, scene, frame_idx)
                ax.set_title(f"Frame: {frame_idx}/{num_frames}")
                return []

            return FuncAnimation(
                fig, update_3d, frames=num_frames, interval=interval, blit=False
            )

        elif view == "2d_topdown":
            fig, ax = plt.subplots(figsize=figsize)

            def update_2d(frame_idx: int) -> list:
                ax.clear()
                self._render_2d_subplot(ax, scene, frame_idx)
                ax.set_title(f"Frame: {frame_idx}/{num_frames}")
                return []

            return FuncAnimation(
                fig, update_2d, frames=num_frames, interval=interval, blit=False
            )

        elif view == "camera":
            fig, ax = plt.subplots(figsize=figsize)

            def update_cam(frame_idx: int) -> list:
                ax.clear()
                self._render_camera_subplot(ax, scene, frame_idx, camera_idx)
                ax.set_title(f"Camera {camera_idx} | Frame: {frame_idx}/{num_frames}")
                return []

            return FuncAnimation(
                fig, update_cam, frames=num_frames, interval=interval, blit=False
            )

        else:
            raise ValueError(
                f"Unknown view type: {view}. Use '3d', '2d_topdown', or 'camera'."
            )

    def _render_3d_subplot(self, ax: Axes3D, scene: Any, frame_idx: int) -> None:
        """Render 3D view on given axes."""
        self.court_renderer.render_3d(ax, show_net=True)

        pos = scene.position[frame_idx]
        x = pos[0] * HALF_DOUBLES_WIDTH
        y = pos[1] * HALF_LENGTH
        z = pos[2] * NET_HEIGHT_POST if len(pos) > 2 else 0

        canonical_pose = scene.canonical_pose_3d[frame_idx]
        sin_yaw = scene.rotation[frame_idx, 0]
        cos_yaw = scene.rotation[frame_idx, 1]

        rot = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]])
        world_pose = canonical_pose @ rot.T
        world_pose[:, 0] += x
        world_pose[:, 1] += y
        world_pose[:, 2] += z

        self.smplh_renderer.render_3d(ax, world_pose)

    def _render_2d_subplot(self, ax: Axes, scene: Any, frame_idx: int) -> None:
        """Render 2D top-down view on given axes."""
        self.court_renderer.render_2d(ax, show_fence=True)

        pos = scene.position[frame_idx]
        x = pos[0] * HALF_DOUBLES_WIDTH
        y = pos[1] * HALF_LENGTH

        # Trail
        start_idx = max(0, frame_idx - 30)
        trail_pos = scene.position[start_idx : frame_idx + 1]
        trail_x = trail_pos[:, 0] * HALF_DOUBLES_WIDTH
        trail_y = trail_pos[:, 1] * HALF_LENGTH
        ax.plot(trail_x, trail_y, "c-", linewidth=2, alpha=0.5)

        ax.scatter([x], [y], c="red", s=100)

        sin_yaw = scene.rotation[frame_idx, 0]
        cos_yaw = scene.rotation[frame_idx, 1]
        ax.arrow(x, y, -sin_yaw, cos_yaw, head_width=0.3, fc="yellow", ec="black")

    def _render_camera_subplot(
        self,
        ax: Axes,
        scene: Any,
        frame_idx: int,
        camera_idx: int,
        *,
        court_kp_size: float = 30.0,
    ) -> None:
        """Render camera view on given axes.
        
        Args:
            ax: Matplotlib axes.
            scene: PLCS scene data.
            frame_idx: Frame index.
            camera_idx: Camera index.
            court_kp_size: Size of court keypoint markers.
        """
        ax.set_facecolor("#1a1a1a")
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)

        cam = scene.cameras[camera_idx]
        human_uv = cam.human_kp_uv[frame_idx]
        human_vis = cam.human_kp_visible[frame_idx]
        court_uv = cam.court_kp_uv[frame_idx]
        court_vis = cam.court_kp_visible[frame_idx]

        visible_court = court_uv[court_vis]
        ax.scatter(visible_court[:, 0], visible_court[:, 1], c="lime", s=court_kp_size, marker="s")

        for i, j in COURT_SKELETON:
            alpha = 0.5 if (court_vis[i] and court_vis[j]) else 0.2
            ax.plot(
                [court_uv[i, 0], court_uv[j, 0]],
                [court_uv[i, 1], court_uv[j, 1]],
                color="lime",
                linewidth=1,
                alpha=alpha,
            )

        self.skeleton_renderer.render_2d(ax, human_uv, human_vis)

        ax.set_title(
            f"Cam {camera_idx} | H:{cam.human_visibility_ratio:.0%} C:{cam.court_visibility_count:.0f}"
        )
