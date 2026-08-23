"""PLCS scene renderer for animation-oriented visualization.

3D views build on the shared rich-rendering primitives in
``src.utils.rendering`` (theme, layers, camera, effects, HUD, minimap); this
renderer owns only the PLCS-specific parts: normalized-position/rotation to
world-pose conversion and HUD line selection. PLCS scenes carry no ball
track, so the HUD shows only the frame clock and the minimap shows only
player position/trail.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from src.tasks.base.visualization.style import SceneStyleConfig
from src.utils.rendering.camera_view import CameraController, apply_scene_camera
from src.utils.rendering.court_renderer import CourtRenderer
from src.utils.rendering.effects import render_fading_line_3d, render_ground_shadow
from src.utils.rendering.hud import HudStyle, format_frame_clock, render_hud_text
from src.utils.rendering.layers import SceneLayer, enable_explicit_layering
from src.utils.rendering.minimap import MinimapRenderer
from src.utils.rendering.skeleton_renderer import SkeletonRenderer, SkeletonStyle
from src.utils.rendering.theme import (
    apply_axes_layout_3d,
    apply_axes_theme_3d,
    apply_figure_theme,
    resolve_theme,
)
from src.utils.schema.court import HALF_DOUBLES_WIDTH, HALF_LENGTH
from src.utils.schema.court_normalization import (
    CourtCoordinateNormalization,
    resolve_court_coordinate_normalization,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    Axes3D: TypeAlias = Any

_VIEW_MARGIN = 2.0
_VIEW_Z_LIMIT = 4.0
_PLAYER_SHADOW_RADIUS = 0.4
_PLAYER_COLOR = "#E76F51"
_GT_COLOR = "green"
_PRED_COLOR = "red"
_PLAYER_COLORS = ("#E76F51", "#2A9D8F", "#E9C46A", "#457B9D")

# Minimap inset rectangle in figure coordinates (left, bottom, width, height).
_MINIMAP_RECT = (0.76, 0.04, 0.21, 0.30)


class PLCSSceneRenderer:
    """Render PLCS scene animations in 3D/top-down/camera views."""

    def __init__(
        self,
        *,
        style: SceneStyleConfig,
        normalization: CourtCoordinateNormalization | str = "v1",
        court_renderer: CourtRenderer | None = None,
        skeleton_renderer: SkeletonRenderer | None = None,
        camera: CameraController | None = None,
    ) -> None:
        self.style = style
        self.normalization = (
            normalization
            if isinstance(normalization, CourtCoordinateNormalization)
            else resolve_court_coordinate_normalization(normalization)
        )
        self.theme = resolve_theme(self.style.theme)
        self.camera = camera or CameraController("broadcast")
        self.court_renderer = court_renderer or CourtRenderer(self.theme.court_style)
        self.skeleton_renderer = skeleton_renderer or SkeletonRenderer(
            skeleton_type="coco17"
        )
        self.smplh_renderer = SkeletonRenderer(skeleton_type="smplh")
        self.coco17_renderer = SkeletonRenderer(skeleton_type="coco17")
        self.hud_style = HudStyle(text_color=self.theme.text_color)
        self.minimap_renderer = MinimapRenderer()

    def _pick_skeleton_renderer(self, num_joints: int) -> SkeletonRenderer:
        """Return the skeleton renderer that matches the joint count."""
        if num_joints == 17:
            return self.coco17_renderer
        return self.smplh_renderer

    @staticmethod
    def _player_scenes(scene: Any) -> list[Any]:
        """Expose each object-axis entry as the existing single-player contract."""
        position = np.asarray(scene.position)
        if position.ndim == 2:
            return [scene]
        num_persons = int(scene.num_persons)
        return [
            SimpleNamespace(
                position=position[:, index],
                rotation=np.asarray(scene.rotation)[:, index],
                canonical_pose_3d=np.asarray(scene.canonical_pose_3d)[:, index],
                meta=scene.meta,
                cameras=scene.cameras,
                num_cameras=scene.num_cameras,
            )
            for index in range(num_persons)
        ]

    def create_animation(
        self,
        scene: Any,
        view: str = "3d",
        camera_idx: int = 0,
        *,
        fps: float,
        figsize: tuple[float, float] = (12, 8),
    ) -> FuncAnimation:
        """Create scene animation for a single view."""
        num_frames = len(scene.position)
        player_scenes = self._player_scenes(scene)
        interval = 1000.0 / fps

        if view == "3d":
            fig = plt.figure(figsize=figsize)
            apply_figure_theme(fig, self.theme)
            ax = fig.add_subplot(111, projection="3d")
            apply_axes_layout_3d(ax, self.theme)
            minimap_ax = (
                fig.add_axes(_MINIMAP_RECT) if self.style.show_minimap else None
            )

            def update_3d(frame_idx: int) -> list[Any]:
                ax.clear()
                self._render_3d_frame(
                    ax,
                    [
                        (
                            player,
                            _PLAYER_COLORS[index % len(_PLAYER_COLORS)],
                            f"Player {index + 1}",
                        )
                        for index, player in enumerate(player_scenes)
                    ],
                    frame_idx,
                    num_frames,
                    fps,
                    title=f"Frame {frame_idx}/{num_frames - 1}",
                )
                if minimap_ax is not None:
                    minimap_ax.clear()
                    self._render_minimap_frame(
                        minimap_ax,
                        [
                            (player, _PLAYER_COLORS[index % len(_PLAYER_COLORS)])
                            for index, player in enumerate(player_scenes)
                        ],
                        frame_idx,
                    )
                return []

            return FuncAnimation(
                fig, update_3d, frames=num_frames, interval=interval, blit=False
            )

        if view == "2d_topdown":
            fig, ax = plt.subplots(figsize=figsize)

            def update_2d(frame_idx: int) -> list[Any]:
                ax.clear()
                for player in player_scenes:
                    self._render_2d_subplot(ax, player, frame_idx)
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
                ax.set_title(
                    f"Camera {camera_idx} | Frame {frame_idx}/{num_frames - 1}"
                )
                return []

            return FuncAnimation(
                fig, update_cam, frames=num_frames, interval=interval, blit=False
            )

        raise ValueError(
            f"Unknown view type: {view}. Use '3d', '2d_topdown', or 'camera'."
        )

    def create_comparison_animation(
        self,
        gt_scene: Any,
        pred_scene: Any,
        view: str = "3d",
        camera_idx: int = 0,
        *,
        fps: float,
        figsize: tuple[float, float] = (12, 8),
        title: str = "GT vs Prediction",
    ) -> FuncAnimation:
        """Create GT vs prediction comparison animation."""
        _ = camera_idx
        gt_frames = len(gt_scene.position)
        pred_frames = len(pred_scene.position)
        num_frames = min(gt_frames, pred_frames)
        if num_frames <= 0:
            raise ValueError("No frames available for comparison animation.")
        interval = 1000.0 / fps

        if view == "3d":
            fig = plt.figure(figsize=figsize)
            apply_figure_theme(fig, self.theme)
            ax = fig.add_subplot(111, projection="3d")
            apply_axes_layout_3d(ax, self.theme)
            minimap_ax = (
                fig.add_axes(_MINIMAP_RECT) if self.style.show_minimap else None
            )

            def update_3d(frame_idx: int) -> list[Any]:
                ax.clear()
                self._render_3d_frame(
                    ax,
                    [
                        (gt_scene, _GT_COLOR, "GT"),
                        (pred_scene, _PRED_COLOR, "Prediction"),
                    ],
                    frame_idx,
                    num_frames,
                    fps,
                    title=f"{title} | Frame {frame_idx}/{num_frames - 1}",
                )
                ax.legend(loc="upper right")
                if minimap_ax is not None:
                    minimap_ax.clear()
                    self._render_minimap_frame(
                        minimap_ax,
                        [(gt_scene, _GT_COLOR), (pred_scene, _PRED_COLOR)],
                        frame_idx,
                    )
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
        print(f"Cameras: {int(scene.num_cameras)}")

        pos = np.asarray(scene.position).reshape(len(scene.position), -1, 3)
        rot = np.asarray(scene.rotation).reshape(len(scene.rotation), -1, 2)
        print("\nPosition (normalized):")
        print(f"  X range: [{pos[..., 0].min():.3f}, {pos[..., 0].max():.3f}]")
        print(f"  Y range: [{pos[..., 1].min():.3f}, {pos[..., 1].max():.3f}]")
        print(f"  Z range: [{pos[..., 2].min():.3f}, {pos[..., 2].max():.3f}]")
        print("Rotation (sin, cos):")
        print(f"  cos range: [{rot[..., 0].min():.3f}, {rot[..., 0].max():.3f}]")
        print(f"  sin range: [{rot[..., 1].min():.3f}, {rot[..., 1].max():.3f}]")
        print("=" * 60)

    def _world_positions(self, scene: Any) -> NDArray[np.float64]:
        """Denormalize the full position track to world court coordinates."""
        pos = np.asarray(scene.position, dtype=np.float64)
        world = self.normalization.denormalize_position(pos)
        return world

    def _render_player_3d(
        self,
        ax: Axes3D,
        scene: Any,
        frame_idx: int,
        *,
        color: str,
        label: str | None,
    ) -> None:
        """Draw one scene's player: movement trail, shadow, and skeleton."""
        world = self._world_positions(scene)
        pos = world[frame_idx]

        if self.style.show_trail:
            trail_start = max(0, frame_idx - self.style.trail_length)
            trail = world[trail_start : frame_idx + 1].astype(np.float32, copy=True)
            if trail.shape[1] > 2:
                trail[:, 2] = 0.02
            render_fading_line_3d(
                ax,
                trail,
                color=color,
                alpha_range=(0.03, 0.5),
                linewidth_range=(1.0, 2.5),
                zorder=SceneLayer.GROUND,
            )

        if self.style.show_shadow and np.isfinite(pos[:2]).all():
            render_ground_shadow(
                ax,
                (float(pos[0]), float(pos[1])),
                radius=_PLAYER_SHADOW_RADIUS,
                alpha=0.28,
                zorder=SceneLayer.GROUND,
            )

        world_pose = self._compute_world_pose(scene, frame_idx)
        skel = self._pick_skeleton_renderer(world_pose.shape[0])
        skel.render_3d(
            ax,
            world_pose,
            label=label,
            style_override=SkeletonStyle(
                joint_color=color,
                bone_color=color,
                joint_size=5.0,
                bone_width=2.0,
            ),
        )

    def _render_3d_frame(
        self,
        ax: Axes3D,
        scenes: list[tuple[Any, str, str | None]],
        frame_idx: int,
        num_frames: int,
        fps: float,
        *,
        title: str,
    ) -> None:
        """Render one 3D frame: court, players, HUD, camera, and title."""
        enable_explicit_layering(ax)
        apply_axes_theme_3d(ax, self.theme)
        x_half_span = float(HALF_DOUBLES_WIDTH + _VIEW_MARGIN)
        y_half_span = float(HALF_LENGTH + _VIEW_MARGIN)
        self.court_renderer.render_3d(
            ax,
            show_net=True,
            apron_bounds=(-x_half_span, x_half_span, -y_half_span, y_half_span),
        )

        for scene, color, label in scenes:
            self._render_player_3d(ax, scene, frame_idx, color=color, label=label)

        if self.style.show_hud:
            render_hud_text(
                ax,
                [format_frame_clock(frame_idx, num_frames - 1, fps)],
                self.hud_style,
            )

        view = self.camera.view_at(frame_idx, fps)
        apply_scene_camera(ax, view, margin=_VIEW_MARGIN, z_limit=_VIEW_Z_LIMIT)
        if self.theme.name != "dark":
            # In dark mode the HUD already shows the frame clock and a title
            # above the full-bleed axes would be clipped anyway.
            ax.set_title(title, color=self.theme.text_color)

    def _render_minimap_frame(
        self,
        minimap_ax: Axes,
        scenes: list[tuple[Any, str]],
        frame_idx: int,
    ) -> None:
        dots = []
        trails = []
        for scene, color in scenes:
            world = self._world_positions(scene)
            trail_start = max(0, frame_idx - self.style.trail_length)
            trails.append(
                (
                    world[trail_start : frame_idx + 1, :2].astype(
                        np.float32, copy=False
                    ),
                    color,
                )
            )
            dots.append(
                ((float(world[frame_idx, 0]), float(world[frame_idx, 1])), color)
            )
        self.minimap_renderer.render(minimap_ax, dots=dots, trails=trails)

    def _render_2d_subplot(self, ax: Axes, scene: Any, frame_idx: int) -> None:
        self.court_renderer.render_2d(ax, show_fence=True)

        world = self._world_positions(scene)
        x = world[frame_idx, 0]
        y = world[frame_idx, 1]

        trail_start = max(0, frame_idx - 30)
        trail = world[trail_start : frame_idx + 1]
        ax.plot(
            trail[:, 0],
            trail[:, 1],
            "c-",
            linewidth=2,
            alpha=0.5,
        )
        ax.scatter([x], [y], c="red", s=100, zorder=5)

        cos_yaw = scene.rotation[frame_idx, 0]
        sin_yaw = scene.rotation[frame_idx, 1]
        ax.arrow(x, y, -sin_yaw, cos_yaw, head_width=0.3, fc="yellow", ec="black")

    def _compute_world_pose(self, scene: Any, frame_idx: int) -> np.ndarray:
        pos = np.asarray(scene.position[frame_idx], dtype=np.float64)
        world_position = self.normalization.denormalize_position(pos)
        x = float(world_position[0])
        y = float(world_position[1])
        z = float(world_position[2])

        canonical_pose = np.asarray(scene.canonical_pose_3d[frame_idx])
        cos_yaw = float(scene.rotation[frame_idx, 0])
        sin_yaw = float(scene.rotation[frame_idx, 1])
        rot = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]])

        world_pose = canonical_pose @ rot.T
        world_pose[:, 0] += x
        world_pose[:, 1] += y
        world_pose[:, 2] += z
        pose: np.ndarray = np.asarray(world_pose, dtype=np.float64)
        return pose

    def _render_2d_comparison_subplot(
        self,
        ax: Axes,
        gt_scene: Any,
        pred_scene: Any,
        frame_idx: int,
    ) -> None:
        self.court_renderer.render_2d(ax, show_fence=True)

        gt_pos = self._world_positions(gt_scene)
        pred_pos = self._world_positions(pred_scene)
        trail_start = max(0, frame_idx - 30)

        gt_trail = gt_pos[trail_start : frame_idx + 1]
        pred_trail = pred_pos[trail_start : frame_idx + 1]
        ax.plot(
            gt_trail[:, 0],
            gt_trail[:, 1],
            color="green",
            linewidth=2,
            alpha=0.7,
            label="GT",
        )
        ax.plot(
            pred_trail[:, 0],
            pred_trail[:, 1],
            color="red",
            linewidth=2,
            alpha=0.7,
            label="Prediction",
        )

        gt_x = gt_pos[frame_idx, 0]
        gt_y = gt_pos[frame_idx, 1]
        pred_x = pred_pos[frame_idx, 0]
        pred_y = pred_pos[frame_idx, 1]
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
        court_vis = cam.court_kp_vis[frame_idx]
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
        human_vis = cam.human_kp_vis[frame_idx]
        if human_uv.ndim == 2:
            self.skeleton_renderer.render_2d(ax, human_uv, human_vis)
            return
        num_persons = int(scene.num_persons)
        for person_index in range(num_persons):
            self.skeleton_renderer.render_2d(
                ax, human_uv[person_index], human_vis[person_index]
            )
