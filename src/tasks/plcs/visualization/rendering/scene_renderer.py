"""PLCS scene renderer for animation-oriented visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from src.utils.rendering.court_renderer import CourtRenderer
from src.utils.rendering.skeleton_renderer import SkeletonRenderer, SkeletonStyle
from src.utils.schema.court import HALF_DOUBLES_WIDTH, HALF_LENGTH, NET_HEIGHT_POST

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    Axes3D: TypeAlias = Any


class PLCSSceneRenderer:
    """Render PLCS scene animations in 3D/top-down/camera views."""

    def __init__(
        self,
        court_renderer: CourtRenderer | None = None,
        skeleton_renderer: SkeletonRenderer | None = None,
    ) -> None:
        self.court_renderer = court_renderer or CourtRenderer()
        self.skeleton_renderer = skeleton_renderer or SkeletonRenderer(
            skeleton_type="coco17"
        )
        self.smplh_renderer = SkeletonRenderer(skeleton_type="smplh")

    def create_animation(
        self,
        scene: Any,
        view: str = "3d",
        camera_idx: int = 0,
        *,
        fps: float = 30.0,
        figsize: tuple[float, float] = (12, 8),
    ) -> FuncAnimation:
        """Create scene animation for a single view."""
        num_frames = int(getattr(scene, "meta", {}).get("num_frames", len(scene.position)))
        interval = 1000.0 / fps

        if view == "3d":
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")

            def update_3d(frame_idx: int) -> list[Any]:
                ax.clear()
                self._render_3d_subplot(ax, scene, frame_idx)
                ax.set_title(f"Frame {frame_idx}/{num_frames - 1}")
                return []

            return FuncAnimation(
                fig, update_3d, frames=num_frames, interval=interval, blit=False
            )

        if view == "2d_topdown":
            fig, ax = plt.subplots(figsize=figsize)

            def update_2d(frame_idx: int) -> list[Any]:
                ax.clear()
                self._render_2d_subplot(ax, scene, frame_idx)
                ax.set_title(f"Top-down | Frame {frame_idx}/{num_frames - 1}")
                return []

            return FuncAnimation(
                fig, update_2d, frames=num_frames, interval=interval, blit=False
            )

        if view == "camera":
            fig, ax = plt.subplots(figsize=figsize)

            def update_cam(frame_idx: int) -> list[Any]:
                ax.clear()
                self._render_camera_subplot(ax, scene, frame_idx, camera_idx)
                ax.set_title(f"Camera {camera_idx} | Frame {frame_idx}/{num_frames - 1}")
                return []

            return FuncAnimation(
                fig, update_cam, frames=num_frames, interval=interval, blit=False
            )

        raise ValueError(f"Unknown view type: {view}. Use '3d', '2d_topdown', or 'camera'.")

    def create_comparison_animation(
        self,
        gt_scene: Any,
        pred_scene: Any,
        view: str = "3d",
        camera_idx: int = 0,
        *,
        fps: float = 30.0,
        figsize: tuple[float, float] = (12, 8),
        title: str = "GT vs Prediction",
    ) -> FuncAnimation:
        """Create GT vs prediction comparison animation."""
        _ = camera_idx
        gt_frames = int(getattr(gt_scene, "meta", {}).get("num_frames", len(gt_scene.position)))
        pred_frames = int(
            getattr(pred_scene, "meta", {}).get("num_frames", len(pred_scene.position))
        )
        num_frames = min(gt_frames, pred_frames)
        if num_frames <= 0:
            raise ValueError("No frames available for comparison animation.")
        interval = 1000.0 / fps

        if view == "3d":
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")

            def update_3d(frame_idx: int) -> list[Any]:
                ax.clear()
                self._render_3d_comparison_subplot(ax, gt_scene, pred_scene, frame_idx)
                ax.set_title(f"{title} | Frame {frame_idx}/{num_frames - 1}")
                return []

            return FuncAnimation(
                fig, update_3d, frames=num_frames, interval=interval, blit=False
            )

        if view == "2d_topdown":
            fig, ax = plt.subplots(figsize=figsize)

            def update_2d(frame_idx: int) -> list[Any]:
                ax.clear()
                self._render_2d_comparison_subplot(ax, gt_scene, pred_scene, frame_idx)
                ax.set_title(f"{title} | Frame {frame_idx}/{num_frames - 1}")
                return []

            return FuncAnimation(
                fig, update_2d, frames=num_frames, interval=interval, blit=False
            )

        if view == "camera":
            raise ValueError(
                "Camera comparison view is not supported for PLCS predict mode. "
                "Use '3d' or '2d_topdown'."
            )

        raise ValueError(
            f"Unknown comparison view type: {view}. Use '3d' or '2d_topdown'."
        )

    def print_scene_info(self, scene: Any) -> None:
        """Print scene metadata and basic statistics."""
        meta = getattr(scene, "meta", {})
        print("=" * 60)
        print("SCENE INFORMATION")
        print("=" * 60)
        print(f"Scene ID: {meta.get('scene_id', 'unknown')}")
        print(f"Motion source: {meta.get('motion_source', 'unknown')}")
        print(f"Category: {meta.get('motion_category', 'unknown')}")
        print(f"Gender: {meta.get('gender', 'unknown')}")
        print(f"Frames: {meta.get('num_frames', 'unknown')}")
        print(f"FPS: {meta.get('fps', 'unknown')}")
        print(f"Cameras: {int(getattr(scene, 'num_cameras', len(scene.cameras)))}")

        pos = np.asarray(scene.position)
        rot = np.asarray(scene.rotation)
        print("\nPosition (normalized):")
        print(f"  X range: [{pos[:, 0].min():.3f}, {pos[:, 0].max():.3f}]")
        print(f"  Y range: [{pos[:, 1].min():.3f}, {pos[:, 1].max():.3f}]")
        print(f"  Z range: [{pos[:, 2].min():.3f}, {pos[:, 2].max():.3f}]")
        print("Rotation (sin, cos):")
        print(f"  sin range: [{rot[:, 0].min():.3f}, {rot[:, 0].max():.3f}]")
        print(f"  cos range: [{rot[:, 1].min():.3f}, {rot[:, 1].max():.3f}]")
        print("=" * 60)

    def _render_3d_subplot(self, ax: Axes3D, scene: Any, frame_idx: int) -> None:
        self.court_renderer.render_3d(ax, show_net=True)
        world_pose = self._compute_world_pose(scene, frame_idx)
        self.smplh_renderer.render_3d(ax, world_pose)
        self._set_zoomed_3d_view(ax, world_pose[:, 0].mean(), world_pose[:, 1].mean())

    def _render_2d_subplot(self, ax: Axes, scene: Any, frame_idx: int) -> None:
        self.court_renderer.render_2d(ax, show_fence=True)

        pos = scene.position
        x = pos[frame_idx, 0] * HALF_DOUBLES_WIDTH
        y = pos[frame_idx, 1] * HALF_LENGTH

        trail_start = max(0, frame_idx - 30)
        trail = pos[trail_start : frame_idx + 1]
        ax.plot(
            trail[:, 0] * HALF_DOUBLES_WIDTH,
            trail[:, 1] * HALF_LENGTH,
            "c-",
            linewidth=2,
            alpha=0.5,
        )
        ax.scatter([x], [y], c="red", s=100, zorder=5)

        cos_yaw = scene.rotation[frame_idx, 0]
        sin_yaw = scene.rotation[frame_idx, 1]
        ax.arrow(x, y, -sin_yaw, cos_yaw, head_width=0.3, fc="yellow", ec="black")

    def _compute_world_pose(self, scene: Any, frame_idx: int) -> np.ndarray:
        pos = scene.position[frame_idx]
        x = float(pos[0]) * HALF_DOUBLES_WIDTH
        y = float(pos[1]) * HALF_LENGTH
        z = float(pos[2]) * NET_HEIGHT_POST if len(pos) > 2 else 0.0

        canonical_pose = np.asarray(scene.canonical_pose_3d[frame_idx])
        cos_yaw = float(scene.rotation[frame_idx, 0])
        sin_yaw = float(scene.rotation[frame_idx, 1])
        rot = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]])

        world_pose = canonical_pose @ rot.T
        world_pose[:, 0] += x
        world_pose[:, 1] += y
        world_pose[:, 2] += z
        return cast(np.ndarray, np.asarray(world_pose, dtype=np.float64))

    def _set_zoomed_3d_view(self, ax: Axes3D, center_x: float, center_y: float) -> None:
        x_half_span = 6.0
        y_half_span = 8.0

        x_min = max(-HALF_DOUBLES_WIDTH - 2.0, center_x - x_half_span)
        x_max = min(HALF_DOUBLES_WIDTH + 2.0, center_x + x_half_span)
        y_min = max(-HALF_LENGTH - 2.0, center_y - y_half_span)
        y_max = min(HALF_LENGTH + 2.0, center_y + y_half_span)

        if x_max - x_min < 2.0:
            x_mid = 0.5 * (x_min + x_max)
            x_min = x_mid - 1.0
            x_max = x_mid + 1.0
        if y_max - y_min < 2.0:
            y_mid = 0.5 * (y_min + y_max)
            y_min = y_mid - 1.0
            y_max = y_mid + 1.0

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(0.0, 3.0)
        ax.set_box_aspect([x_max - x_min, y_max - y_min, 3.0])

    def _render_3d_comparison_subplot(
        self,
        ax: Axes3D,
        gt_scene: Any,
        pred_scene: Any,
        frame_idx: int,
    ) -> None:
        self.court_renderer.render_3d(ax, show_net=True)

        gt_pose = self._compute_world_pose(gt_scene, frame_idx)
        pred_pose = self._compute_world_pose(pred_scene, frame_idx)

        self.smplh_renderer.render_3d(
            ax,
            gt_pose,
            label="GT",
            style_override=SkeletonStyle(
                joint_color="green",
                bone_color="green",
                joint_size=5.0,
                bone_width=2.0,
            ),
        )
        self.smplh_renderer.render_3d(
            ax,
            pred_pose,
            label="Prediction",
            style_override=SkeletonStyle(
                joint_color="red",
                bone_color="red",
                joint_size=5.0,
                bone_width=2.0,
            ),
        )

        gt_xy = gt_pose[:, :2].mean(axis=0)
        pred_xy = pred_pose[:, :2].mean(axis=0)
        center_x = float(0.5 * (gt_xy[0] + pred_xy[0]))
        center_y = float(0.5 * (gt_xy[1] + pred_xy[1]))
        self._set_zoomed_3d_view(ax, center_x, center_y)
        ax.legend(loc="upper right")

    def _render_2d_comparison_subplot(
        self,
        ax: Axes,
        gt_scene: Any,
        pred_scene: Any,
        frame_idx: int,
    ) -> None:
        self.court_renderer.render_2d(ax, show_fence=True)

        gt_pos = np.asarray(gt_scene.position)
        pred_pos = np.asarray(pred_scene.position)
        trail_start = max(0, frame_idx - 30)

        gt_trail = gt_pos[trail_start : frame_idx + 1]
        pred_trail = pred_pos[trail_start : frame_idx + 1]
        ax.plot(
            gt_trail[:, 0] * HALF_DOUBLES_WIDTH,
            gt_trail[:, 1] * HALF_LENGTH,
            color="green",
            linewidth=2,
            alpha=0.7,
            label="GT",
        )
        ax.plot(
            pred_trail[:, 0] * HALF_DOUBLES_WIDTH,
            pred_trail[:, 1] * HALF_LENGTH,
            color="red",
            linewidth=2,
            alpha=0.7,
            label="Prediction",
        )

        gt_x = gt_pos[frame_idx, 0] * HALF_DOUBLES_WIDTH
        gt_y = gt_pos[frame_idx, 1] * HALF_LENGTH
        pred_x = pred_pos[frame_idx, 0] * HALF_DOUBLES_WIDTH
        pred_y = pred_pos[frame_idx, 1] * HALF_LENGTH
        ax.scatter([gt_x], [gt_y], c="green", s=80, zorder=6)
        ax.scatter([pred_x], [pred_y], c="red", s=80, zorder=6)

        gt_cos = gt_scene.rotation[frame_idx, 0]
        gt_sin = gt_scene.rotation[frame_idx, 1]
        pred_cos = pred_scene.rotation[frame_idx, 0]
        pred_sin = pred_scene.rotation[frame_idx, 1]
        ax.arrow(gt_x, gt_y, -gt_sin, gt_cos, head_width=0.25, fc="green", ec="green")
        ax.arrow(
            pred_x,
            pred_y,
            -pred_sin,
            pred_cos,
            head_width=0.25,
            fc="red",
            ec="red",
        )
        ax.legend(loc="upper right")

    def _render_camera_subplot(
        self,
        ax: Axes,
        scene: Any,
        frame_idx: int,
        camera_idx: int,
    ) -> None:
        if camera_idx < 0 or camera_idx >= len(scene.cameras):
            raise ValueError(
                f"Camera index {camera_idx} out of range (0-{len(scene.cameras) - 1})."
            )

        cam = scene.cameras[camera_idx]
        ax.set_facecolor("#1a1a1a")
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)

        court_uv = cam.court_kp_uv[frame_idx]
        court_vis = cam.court_kp_visible[frame_idx]
        self.court_renderer.render_projected_2d(
            ax,
            court_uv,
            court_vis,
            line_color="lime",
            line_width=1.0,
            visible_line_alpha=0.5,
            partial_line_alpha=0.2,
            keypoint_color="lime",
            keypoint_size=30.0,
            keypoint_alpha=0.7,
            keypoint_marker="s",
        )

        human_uv = cam.human_kp_uv[frame_idx]
        human_vis = cam.human_kp_visible[frame_idx]
        self.skeleton_renderer.render_2d(ax, human_uv, human_vis)
