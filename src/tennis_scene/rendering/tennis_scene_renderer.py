"""Tennis scene renderer for 3D animation visualization.

This module renders complete tennis scenes in 3D, including:
- Court geometry (two-tone surface, meshed net, posts)
- Multi-player body representation (SMPL mesh or 3D skeleton), with ground
  shadows and fading movement trails
- Ball with fading trajectory trail, ground shadow, and bounce-impact rings
- Virtual camera work (presets / orbit / keyframes via ``CameraController``)
- HUD overlay (frame clock, ball speed, bounce count) and a top-down minimap

The drawing primitives (camera, theme, layers, HUD, minimap, effects) live in
``src.utils.rendering``; this renderer owns only the ``SceneResult``-specific
adaptation: SMPL-to-court transforms, per-player colors, HUD line selection,
and minimap array extraction.

Only 3D rendering is supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from src.submodules.vendor.gvhmr.body_model import load_smpl_faces
from src.utils.configuration import SemanticConfigurationError
from src.utils.geometry.matrices import (
    axis_angle_to_rotation_matrix,
    rotation_matrix_z,
    smpl_y_up_to_court_z_up,
)
from src.utils.rendering.ball_renderer import BallRenderer, BallStyle
from src.utils.rendering.camera_view import (
    DEFAULT_VIEW_Z_LIMIT,
    CameraController,
    apply_scene_camera,
)
from src.utils.rendering.court_renderer import CourtRenderer, CourtStyle
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
from src.utils.rendering.mesh_renderer import MeshRenderer
from src.utils.rendering.minimap import MinimapRenderer, MinimapStyle
from src.utils.rendering.skeleton_renderer import SkeletonRenderer, SkeletonStyle
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
    from mpl_toolkits.mplot3d import Axes3D
    from numpy.typing import NDArray

    from src.tennis_scene.io import SceneResult

_FIXED_VIEW_MARGIN = 2.0

_DEFAULT_PLAYER_COLORS = [
    "#E76F51",
    "#2A9D8F",
    "#E9C46A",
    "#264653",
    "#F4A261",
    "#5E81AC",
]

_PLAYER_SHADOW_RADIUS = 0.4
_BALL_SHADOW_RADIUS = 0.08
_BOUNCE_RING_COLOR = "#FFD700"
_MINIMAP_BALL_TRAIL_FRAMES = 30

# Minimap inset rectangle in figure coordinates (left, bottom, width, height).
_MINIMAP_RECT = (0.76, 0.04, 0.21, 0.30)


@dataclass
class TennisSceneStyle:
    """Style configuration for 3D tennis scene rendering."""

    court_style: CourtStyle | None = None
    ball_style: BallStyle | None = None
    skeleton_style: SkeletonStyle | None = None
    trail_length: int = 30
    show_trail: bool = True
    figsize: tuple[float, float] = (12, 8)
    mesh_color: str = "#7BC8F6"
    mesh_alpha: float = 0.6
    player_representation: Literal["smpl", "skeleton"] = "skeleton"

    # Scene look & feel
    theme: Literal["light", "dark"] = "light"
    show_ball_shadow: bool = True
    show_player_shadow: bool = True
    show_player_trail: bool = True
    player_trail_length: int = 60
    show_bounces: bool = True
    bounce_marker_duration_s: float = 1.5

    # Overlays
    show_hud: bool = True
    hud_style: HudStyle | None = None
    show_minimap: bool = True
    minimap_style: MinimapStyle | None = None


class TennisSceneRenderer:
    """Render complete 3D tennis scenes with animation support."""

    def __init__(
        self,
        style: TennisSceneStyle,
        *,
        smpl_faces_path: Path,
        smpl_joint_regressor_path: Path,
        camera: CameraController,
    ) -> None:
        self.style = style
        self.camera = camera
        self.theme = resolve_theme(self.style.theme)
        court_style = self.style.court_style
        if court_style is None:
            court_style = self.theme.court_style
        self.court_renderer = CourtRenderer(court_style)
        self.ball_renderer = BallRenderer(self.style.ball_style)
        self.skeleton_renderer = SkeletonRenderer(
            skeleton_type="smpl",
            style=self.style.skeleton_style,
        )
        self.hud_style = self.style.hud_style
        if self.hud_style is None:
            self.hud_style = HudStyle(text_color=self.theme.text_color)
        self.minimap_renderer = MinimapRenderer(self.style.minimap_style)

        self._mesh_renderer: MeshRenderer | None = None
        self._smpl_joint_regressor: NDArray[np.float32] | None = None
        self._scene_vertices_cache: dict[int, NDArray[np.float32]] = {}
        self._scene_joints_cache: dict[int, NDArray[np.float32]] = {}
        self._scene_ball_speeds_cache: dict[int, NDArray[np.float32]] = {}
        self._scene_bounce_frames_cache: dict[int, NDArray[np.int64]] = {}

        faces_path = self._require_resolved_asset_path(
            smpl_faces_path,
            name="SMPL faces body-model asset",
        )
        regressor_path = self._require_resolved_asset_path(
            smpl_joint_regressor_path,
            name="SMPL joint-regressor asset",
        )

        # Validate both configured assets for every representation. Skeleton mode
        # still derives joints from SMPL vertices, and must not silently accept an
        # invalid faces/body-model path merely because it does not draw the mesh.
        faces = load_smpl_faces(faces_path)
        self._smpl_joint_regressor = self._load_smpl_joint_regressor(regressor_path)

        if self.style.player_representation == "smpl":
            self._mesh_renderer = MeshRenderer(faces)

    @staticmethod
    def _require_resolved_asset_path(path: Path, *, name: str) -> Path:
        if not isinstance(path, Path) or not path.is_absolute():
            raise SemanticConfigurationError(
                f"{name} must be an absolute pathlib.Path resolved by "
                f"PathResolver; got {path!r}."
            )
        resolved = path.resolve(strict=False)
        if path != resolved:
            raise SemanticConfigurationError(
                f"{name} must already be normalized by PathResolver; got {path!r}."
            )
        if not resolved.is_file():
            raise FileNotFoundError(f"{name} not found: {resolved}.")
        return resolved

    def _load_smpl_joint_regressor(
        self,
        regressor_path: Path,
    ) -> NDArray[np.float32]:
        import torch

        reg = torch.load(regressor_path, map_location="cpu", weights_only=False)
        if isinstance(reg, torch.Tensor):
            result = reg.cpu().numpy().astype(np.float32)
        else:
            result = np.asarray(reg, dtype=np.float32)
        if result.ndim != 2:
            raise ValueError(
                "SMPL joint-regressor asset must contain a 2D (joints, vertices) "
                f"array, got shape {result.shape}."
            )
        return result

    def _player_color(self, player_idx: int) -> str:
        return _DEFAULT_PLAYER_COLORS[player_idx % len(_DEFAULT_PLAYER_COLORS)]

    def _get_player_tracks(self, scene: SceneResult) -> list[int]:
        if scene.player_track_ids is None:
            raise RuntimeError("scene.player_track_ids is required for 3D rendering.")
        return [int(track_id) for track_id in scene.player_track_ids.tolist()]

    def _get_players_position(self, scene: SceneResult) -> NDArray[np.float32]:
        return scene.player_position

    def _axis_angle_to_matrix(
        self, axis_angle: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        return axis_angle_to_rotation_matrix(axis_angle)

    def _rotation_matrix_z(self, yaw: NDArray[np.float32]) -> NDArray[np.float32]:
        return rotation_matrix_z(yaw)

    def _validate_required_smpl_fields(self, scene: SceneResult) -> None:
        missing: list[str] = []
        if scene.smpl_vertices_local is None:
            missing.append("smpl_vertices_local")
        if missing:
            missing_str = ", ".join(missing)
            raise RuntimeError(
                "SMPL rendering requires the following SceneResult fields: "
                f"{missing_str}."
            )

    def _build_players_smpl_vertices_court(
        self, scene: SceneResult
    ) -> NDArray[np.float32]:
        cache_key = id(scene)
        cached = self._scene_vertices_cache.get(cache_key)
        if cached is not None:
            return cached

        self._validate_required_smpl_fields(scene)
        if self._smpl_joint_regressor is None:
            raise RuntimeError("SMPL joint regressor is not loaded.")

        verts_local = np.asarray(scene.smpl_vertices_local, dtype=np.float32)
        global_orient = np.asarray(scene.smpl_global_orient, dtype=np.float32)
        players_position = np.asarray(scene.player_position, dtype=np.float32)
        players_yaw = np.asarray(scene.player_yaw, dtype=np.float32)

        if verts_local.ndim != 4 or verts_local.shape[-1] != 3:
            raise RuntimeError(
                "smpl_vertices_local must have shape (P, T, V, 3), "
                f"got {verts_local.shape}."
            )
        if global_orient.shape != verts_local.shape[:2] + (3,):
            raise RuntimeError(
                "smpl_global_orient must have shape (P, T, 3), "
                f"got {global_orient.shape}."
            )
        if players_position.shape != verts_local.shape[:2] + (3,):
            raise RuntimeError(
                "player_position must have shape (P, T, 3), "
                f"got {players_position.shape}."
            )
        if players_yaw.shape != verts_local.shape[:2]:
            raise RuntimeError(
                f"player_yaw must have shape (P, T), got {players_yaw.shape}."
            )

        roots = np.einsum("jv,ptvc->ptjc", self._smpl_joint_regressor, verts_local)[
            :, :, 0, :
        ]
        verts_centered = verts_local - roots[:, :, None, :]

        orient_rot = self._axis_angle_to_matrix(global_orient)
        verts_pose_local = np.einsum("ptji,ptvj->ptvi", orient_rot, verts_centered)

        verts_court_local = smpl_y_up_to_court_z_up(verts_pose_local)
        plcs_rot = self._rotation_matrix_z(players_yaw)
        verts_court = np.einsum("ptij,ptvj->ptvi", plcs_rot, verts_court_local)
        verts_court = verts_court + players_position[:, :, None, :]
        verts_court = cast("NDArray[np.float32]", verts_court.astype(np.float32))

        self._scene_vertices_cache[cache_key] = verts_court
        return verts_court

    def _get_players_kp_3d(self, scene: SceneResult) -> NDArray[np.float32]:
        cache_key = id(scene)
        cached = self._scene_joints_cache.get(cache_key)
        if cached is not None:
            return cached

        if self._smpl_joint_regressor is None:
            raise RuntimeError("SMPL joint regressor is not loaded.")

        players_smpl = self._build_players_smpl_vertices_court(scene)
        players_kp_3d = cast(
            "NDArray[np.float32]",
            np.einsum("jv,ptvc->ptjc", self._smpl_joint_regressor, players_smpl).astype(
                np.float32
            ),
        )
        self._scene_joints_cache[cache_key] = players_kp_3d
        return players_kp_3d

    def _get_ball_speeds(self, scene: SceneResult) -> NDArray[np.float32] | None:
        """Per-frame ball speed (m/s), cached per scene. None without ball_3d."""
        if scene.ball_3d is None:
            return None
        cache_key = id(scene)
        cached = self._scene_ball_speeds_cache.get(cache_key)
        if cached is None:
            cached = compute_speeds(scene.ball_3d, scene.fps)
            self._scene_ball_speeds_cache[cache_key] = cached
        return cached

    def _get_bounce_frames(self, scene: SceneResult) -> NDArray[np.int64] | None:
        """Ball bounce frame indices, cached per scene. None without ball_3d."""
        if scene.ball_3d is None:
            return None
        cache_key = id(scene)
        cached = self._scene_bounce_frames_cache.get(cache_key)
        if cached is None:
            cached = detect_bounces(scene.ball_3d)
            self._scene_bounce_frames_cache[cache_key] = cached
        return cached

    def render_frame_3d(
        self,
        scene: SceneResult,
        frame_idx: int,
        *,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        ax: Axes3D | None = None,
    ) -> tuple[Figure | None, Axes3D]:
        """Render a single frame in 3D.

        When ``ax`` is provided the renderer draws only into it (no minimap
        inset, no figure-level styling); otherwise a new figure is created
        with the full overlay set.
        """
        fig = None
        if ax is None:
            selected_figsize = self.style.figsize if figsize is None else figsize
            fig = plt.figure(figsize=selected_figsize)
            apply_figure_theme(fig, self.theme)
            ax = fig.add_subplot(111, projection="3d")
            apply_axes_layout_3d(ax, self.theme)

        self._render_3d_internal(ax, scene, frame_idx)
        if title is not None:
            ax.set_title(title, color=self.theme.text_color)

        if fig is not None and self.style.show_minimap:
            minimap_ax = self._add_minimap_axes(fig)
            self._render_minimap(minimap_ax, scene, frame_idx)
        return fig, ax

    def create_animation(
        self,
        scene: SceneResult,
        *,
        fps: float | None = None,
        figsize: tuple[float, float] | None = None,
        start_frame: int = 0,
        end_frame: int | None = None,
    ) -> FuncAnimation:
        """Create 3D animation of scene."""
        if fps is None:
            fps = scene.fps
        if end_frame is None:
            end_frame = scene.num_frames

        selected_figsize = self.style.figsize if figsize is None else figsize
        fig = plt.figure(figsize=selected_figsize)
        apply_figure_theme(fig, self.theme)
        ax = fig.add_subplot(111, projection="3d")
        apply_axes_layout_3d(ax, self.theme)
        minimap_ax = self._add_minimap_axes(fig) if self.style.show_minimap else None
        interval = 1000.0 / fps
        frames_range = range(start_frame, end_frame)

        def update(frame_idx: int) -> list:
            ax.clear()
            self._render_3d_internal(ax, scene, frame_idx)
            if minimap_ax is not None:
                minimap_ax.clear()
                self._render_minimap(minimap_ax, scene, frame_idx)
            return []

        return FuncAnimation(
            fig, update, frames=frames_range, interval=interval, blit=False
        )

    def save_animation(
        self,
        scene: SceneResult,
        output_path: Path,
        *,
        fps: float | None = None,
        figsize: tuple[float, float] | None = None,
        start_frame: int = 0,
        end_frame: int | None = None,
        dpi: int = 100,
        writer: str = "ffmpeg",
    ) -> None:
        """Save 3D animation as video file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        selected_fps = scene.fps if fps is None else fps

        anim = self.create_animation(
            scene,
            fps=selected_fps,
            figsize=figsize,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        anim.save(
            str(output_path), writer=writer, fps=int(round(selected_fps)), dpi=dpi
        )

    def _add_minimap_axes(self, fig: Figure) -> Axes:
        return fig.add_axes(_MINIMAP_RECT)

    def _render_minimap(
        self, minimap_ax: Axes, scene: SceneResult, frame_idx: int
    ) -> None:
        """Extract plain arrays for the current frame and draw the minimap."""
        dots: list[tuple[tuple[float, float], str]] = []
        for player_idx in range(scene.player_position.shape[0]):
            pos = scene.player_position[player_idx, frame_idx]
            dots.append(
                ((float(pos[0]), float(pos[1])), self._player_color(player_idx))
            )

        trails: list[tuple[NDArray[np.float32], str]] = []
        trail_dots: list[tuple[tuple[float, float], str]] = []
        event_marks_xy: NDArray[np.float32] | None = None
        if scene.ball_3d is not None:
            ball_color = self.ball_renderer.style.ball_color
            bounce_frames = self._get_bounce_frames(scene)
            if bounce_frames is not None:
                past = bounce_frames[bounce_frames <= frame_idx]
                event_marks_xy = scene.ball_3d[past, :2]

            trail_start = max(0, frame_idx - _MINIMAP_BALL_TRAIL_FRAMES)
            trails.append((scene.ball_3d[trail_start : frame_idx + 1, :2], ball_color))
            ball_pos = scene.ball_3d[frame_idx]
            trail_dots.append(((float(ball_pos[0]), float(ball_pos[1])), ball_color))

        self.minimap_renderer.render(
            minimap_ax,
            dots=dots,
            trails=trails,
            trail_dots=trail_dots,
            event_marks_xy=event_marks_xy,
        )

    def _render_bounce_rings(
        self, ax: Axes3D, scene: SceneResult, frame_idx: int
    ) -> None:
        bounce_frames = self._get_bounce_frames(scene)
        if bounce_frames is None or scene.ball_3d is None:
            return
        duration_frames = max(
            1, int(round(self.style.bounce_marker_duration_s * scene.fps))
        )
        for b in bounce_frames.tolist():
            age_frames = frame_idx - b
            if age_frames < 0 or age_frames > duration_frames:
                continue
            pos = scene.ball_3d[b]
            if not np.isfinite(pos[:2]).all():
                continue
            render_impact_ring(
                ax,
                (float(pos[0]), float(pos[1])),
                age_frames / duration_frames,
                color=_BOUNCE_RING_COLOR,
                zorder=SceneLayer.RING,
            )

    def _render_players(self, ax: Axes3D, scene: SceneResult, frame_idx: int) -> None:
        track_ids = self._get_player_tracks(scene)
        players_position = self._get_players_position(scene)
        players_smpl = self._build_players_smpl_vertices_court(scene)

        for player_idx, track_id in enumerate(track_ids):
            color = self._player_color(player_idx)
            pos = players_position[player_idx, frame_idx]

            if self.style.show_player_trail:
                trail_start = max(0, frame_idx - self.style.player_trail_length)
                trail = players_position[player_idx, trail_start : frame_idx + 1].copy()
                trail[:, 2] = 0.02
                render_fading_line_3d(
                    ax,
                    trail,
                    color=color,
                    alpha_range=(0.03, 0.5),
                    linewidth_range=(1.0, 2.5),
                    zorder=SceneLayer.GROUND,
                )

            if self.style.show_player_shadow and np.isfinite(pos[:2]).all():
                render_ground_shadow(
                    ax,
                    (float(pos[0]), float(pos[1])),
                    radius=_PLAYER_SHADOW_RADIUS,
                    alpha=0.28,
                    zorder=SceneLayer.GROUND,
                )

            if self.style.player_representation == "smpl":
                if self._mesh_renderer is None:
                    raise RuntimeError("SMPL mesh renderer is not initialized")
                self._mesh_renderer.render_3d(
                    ax,
                    players_smpl[player_idx, frame_idx],
                    color=color,
                    alpha=self.style.mesh_alpha,
                    zorder=SceneLayer.PLAYER,
                )
            else:
                style_override = SkeletonStyle(
                    joint_color=color,
                    bone_color=color,
                    joint_size=(
                        self.style.skeleton_style.joint_size
                        if self.style.skeleton_style
                        else 5.0
                    ),
                    bone_width=(
                        self.style.skeleton_style.bone_width
                        if self.style.skeleton_style
                        else 2.0
                    ),
                    joint_alpha=(
                        self.style.skeleton_style.joint_alpha
                        if self.style.skeleton_style
                        else 1.0
                    ),
                    bone_alpha=(
                        self.style.skeleton_style.bone_alpha
                        if self.style.skeleton_style
                        else 0.8
                    ),
                )
                self.skeleton_renderer.render_3d(
                    ax,
                    self._get_players_kp_3d(scene)[player_idx, frame_idx],
                    style_override=style_override,
                )

            ax.text(
                pos[0],
                pos[1],
                pos[2] + 1.0,
                f"P{track_id}",
                color=color,
            )

    def _render_ball(self, ax: Axes3D, scene: SceneResult, frame_idx: int) -> None:
        if scene.ball_3d is None:
            return

        ball_pos = scene.ball_3d[frame_idx]
        ball_valid = bool(np.isfinite(ball_pos).all())

        if self.style.show_trail:
            start_idx = max(0, frame_idx - self.style.trail_length)
            trail = scene.ball_3d[start_idx : frame_idx + 1]
            render_fading_line_3d(
                ax,
                trail,
                color=self.ball_renderer.style.trajectory_color,
                alpha_range=(0.05, 0.95),
                linewidth_range=(1.0, 3.0),
                zorder=SceneLayer.TRAIL,
            )

        if ball_valid:
            if self.style.show_ball_shadow:
                # Fade the contact shadow out as the ball rises.
                height_ratio = float(
                    np.clip(ball_pos[2] / DEFAULT_VIEW_Z_LIMIT, 0.0, 1.0)
                )
                render_ground_shadow(
                    ax,
                    (float(ball_pos[0]), float(ball_pos[1])),
                    radius=_BALL_SHADOW_RADIUS,
                    alpha=0.35 * (1.0 - 0.7 * height_ratio),
                    zorder=SceneLayer.GROUND,
                )
            self.ball_renderer.render_ball_3d(
                ax, ball_pos, label="Ball", zorder=SceneLayer.BALL
            )

    def _render_hud(self, ax: Axes3D, scene: SceneResult, frame_idx: int) -> None:
        """Select the HUD lines this scene supports and draw them."""
        if not self.style.show_hud:
            return
        lines = [format_frame_clock(frame_idx, scene.num_frames, scene.fps)]
        speeds = self._get_ball_speeds(scene)
        if speeds is not None:
            lines.append(f"Ball speed {format_speed_kmh(float(speeds[frame_idx]))}")
        bounce_frames = self._get_bounce_frames(scene)
        if bounce_frames is not None:
            lines.append(f"Bounces {int((bounce_frames <= frame_idx).sum())}")
        render_hud_text(ax, lines, self.hud_style)

    def _render_3d_internal(
        self, ax: Axes3D, scene: SceneResult, frame_idx: int
    ) -> None:
        enable_explicit_layering(ax)
        apply_axes_theme_3d(ax, self.theme)

        x_half_span = float(HALF_DOUBLES_WIDTH + _FIXED_VIEW_MARGIN)
        y_half_span = float(HALF_LENGTH + _FIXED_VIEW_MARGIN)
        self.court_renderer.render_3d(
            ax,
            show_net=True,
            apron_bounds=(-x_half_span, x_half_span, -y_half_span, y_half_span),
        )

        if self.style.show_bounces:
            self._render_bounce_rings(ax, scene, frame_idx)
        self._render_players(ax, scene, frame_idx)
        self._render_ball(ax, scene, frame_idx)
        self._render_hud(ax, scene, frame_idx)

        view = self.camera.view_at(frame_idx, scene.fps)
        apply_scene_camera(ax, view, margin=_FIXED_VIEW_MARGIN)
        if self.theme.name != "dark":
            # In dark mode the HUD already shows the frame clock and a title
            # above the full-bleed axes would be clipped anyway.
            ax.set_title(
                f"Frame: {frame_idx}/{scene.num_frames}", color=self.theme.text_color
            )
