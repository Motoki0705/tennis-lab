"""Tennis scene renderer for 3D animation visualization.

This module renders complete tennis scenes in 3D, including:
- Court geometry
- Multi-player body representation (SMPL mesh or 3D skeleton)
- Ball trajectory

Only 3D rendering is supported.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.utils.geometry.matrices import (
    axis_angle_to_rotation_matrix,
    rotation_matrix_z,
)
from src.utils.rendering.ball_renderer import BallRenderer, BallStyle
from src.utils.rendering.court_renderer import CourtRenderer, CourtStyle
from src.utils.rendering.skeleton_renderer import SkeletonRenderer, SkeletonStyle

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D
    from numpy.typing import NDArray

    from src.tennis_scene.io import SceneResult

DEFAULT_SMPL_MODEL_PATH = Path("data/smplh/neutral/model.npz")
DEFAULT_SMPL_JOINT_REGRESSOR_PATH = Path(
    "third_party/GVHMR/hmr4d/utils/body_model/smpl_neutral_J_regressor.pt"
)

_DEFAULT_PLAYER_COLORS = [
    "#E76F51",
    "#2A9D8F",
    "#E9C46A",
    "#264653",
    "#F4A261",
    "#5E81AC",
]


@dataclass
class TennisSceneStyle:
    """Style configuration for 3D tennis scene rendering."""

    court_style: CourtStyle | None = None
    ball_style: BallStyle | None = None
    skeleton_style: SkeletonStyle | None = None
    trail_length: int = 30
    show_direction: bool = True
    show_trail: bool = True
    figsize: tuple[float, float] = (12, 8)
    mesh_color: str = "#7BC8F6"
    mesh_alpha: float = 0.6
    player_representation: Literal["smpl", "skeleton"] = "skeleton"


class TennisSceneRenderer:
    """Render complete 3D tennis scenes with animation support."""

    def __init__(
        self,
        style: TennisSceneStyle | None = None,
        smpl_model_path: str | Path | None = None,
        smpl_joint_regressor_path: str | Path | None = None,
    ) -> None:
        self.style = style or TennisSceneStyle()
        self.court_renderer = CourtRenderer(self.style.court_style)
        self.ball_renderer = BallRenderer(self.style.ball_style)
        self.skeleton_renderer = SkeletonRenderer(
            skeleton_type="smpl",
            style=self.style.skeleton_style,
        )

        self._smpl_faces: NDArray[np.int64] | None = None
        self._smpl_joint_regressor: NDArray[np.float32] | None = None
        self._scene_vertices_cache: dict[int, NDArray[np.float32]] = {}
        self._scene_joints_cache: dict[int, NDArray[np.float32]] = {}

        regressor_path = smpl_joint_regressor_path or DEFAULT_SMPL_JOINT_REGRESSOR_PATH
        self._smpl_joint_regressor = self._load_smpl_joint_regressor(regressor_path)
        if self._smpl_joint_regressor is None:
            raise RuntimeError(
                f"Failed to load SMPL joint regressor from {regressor_path}. "
                "SMPL rendering requires this file."
            )

        if self.style.player_representation == "smpl":
            model_path = smpl_model_path or DEFAULT_SMPL_MODEL_PATH
            self._smpl_faces = self._load_smpl_faces(model_path)
            if self._smpl_faces is None:
                raise RuntimeError(
                    f"Failed to load SMPL faces from {model_path}. "
                    "SMPL rendering is unavailable in this environment."
                )

    def _load_smpl_faces(self, model_path: str | Path) -> NDArray[np.int64] | None:
        model_path = Path(model_path)
        if not model_path.exists():
            return None
        try:
            if model_path.suffix == ".npz":
                data = np.load(model_path)
                return data["f"].astype(np.int64)
            if model_path.suffix == ".pkl":
                with model_path.open("rb") as f:
                    data = pickle.load(f, encoding="latin1")
                return np.asarray(data["f"], dtype=np.int64)
        except Exception:
            return None
        return None

    def _load_smpl_joint_regressor(
        self,
        regressor_path: str | Path,
    ) -> NDArray[np.float32] | None:
        path = Path(regressor_path)
        if not path.exists():
            return None
        try:
            import torch

            reg = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(reg, torch.Tensor):
                return reg.cpu().numpy().astype(np.float32)
            return np.asarray(reg, dtype=np.float32)
        except Exception:
            return None

    def _player_color(self, player_idx: int) -> str:
        return _DEFAULT_PLAYER_COLORS[player_idx % len(_DEFAULT_PLAYER_COLORS)]

    def _get_player_tracks(self, scene: SceneResult) -> list[int]:
        if scene.player_track_ids is None:
            raise RuntimeError("scene.player_track_ids is required for 3D rendering.")
        return [int(track_id) for track_id in scene.player_track_ids.tolist()]

    def _get_players_position(self, scene: SceneResult) -> NDArray[np.float32]:
        return scene.player_position

    def _get_players_yaw(self, scene: SceneResult) -> NDArray[np.float32]:
        return scene.player_yaw

    def _axis_angle_to_matrix(self, axis_angle: NDArray[np.float32]) -> NDArray[np.float32]:
        return axis_angle_to_rotation_matrix(axis_angle)

    def _rotation_matrix_z(self, yaw: NDArray[np.float32]) -> NDArray[np.float32]:
        return rotation_matrix_z(yaw)

    def _validate_required_smpl_fields(self, scene: SceneResult) -> None:
        missing: list[str] = []
        if scene.smpl_vertices_local is None:
            missing.append("smpl_vertices_local")
        if scene.smpl_global_orient is None:
            missing.append("smpl_global_orient")
        if scene.player_position is None:
            missing.append("player_position")
        if scene.player_yaw is None:
            missing.append("player_yaw")
        if missing:
            missing_str = ", ".join(missing)
            raise RuntimeError(
                "SMPL rendering requires the following SceneResult fields: "
                f"{missing_str}."
            )

    def _build_players_smpl_vertices_court(self, scene: SceneResult) -> NDArray[np.float32]:
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
                "player_yaw must have shape (P, T), "
                f"got {players_yaw.shape}."
            )

        roots = np.einsum("jv,ptvc->ptjc", self._smpl_joint_regressor, verts_local)[:, :, 0, :]
        verts_centered = verts_local - roots[:, :, None, :]

        orient_rot = self._axis_angle_to_matrix(global_orient)
        verts_pose_local = np.einsum("ptji,ptvj->ptvi", orient_rot, verts_centered)

        plcs_rot = self._rotation_matrix_z(players_yaw)
        verts_court = np.einsum("ptij,ptvj->ptvi", plcs_rot, verts_pose_local)
        verts_court = verts_court + players_position[:, :, None, :]
        verts_court = verts_court.astype(np.float32)

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
        players_kp_3d = np.einsum("jv,ptvc->ptjc", self._smpl_joint_regressor, players_smpl).astype(
            np.float32
        )
        self._scene_joints_cache[cache_key] = players_kp_3d
        return players_kp_3d

    def _render_smpl_mesh_3d(
        self,
        ax: Axes3D,
        vertices: NDArray[np.float32],
        color: str,
        alpha: float,
    ) -> None:
        if self._smpl_faces is None:
            raise RuntimeError("SMPL faces are not loaded")
        triangles = vertices[self._smpl_faces]
        mesh = Poly3DCollection(
            triangles,
            alpha=alpha,
            facecolor=color,
            edgecolor="none",
            linewidths=0.0,
        )
        ax.add_collection3d(mesh)

    def render_frame_3d(
        self,
        scene: SceneResult,
        frame_idx: int,
        *,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        ax: Axes3D | None = None,
    ) -> tuple[Figure | None, Axes3D]:
        """Render a single frame in 3D."""
        fig = None
        if ax is None:
            fig = plt.figure(figsize=figsize or self.style.figsize)
            ax = fig.add_subplot(111, projection="3d")

        self._render_3d_internal(ax, scene, frame_idx)
        if title is None:
            title = f"Frame: {frame_idx}/{scene.num_frames}"
        ax.set_title(title)
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

        fig = plt.figure(figsize=figsize or self.style.figsize)
        ax = fig.add_subplot(111, projection="3d")
        interval = 1000.0 / fps
        frames_range = range(start_frame, end_frame)

        def update(frame_idx: int) -> list:
            ax.clear()
            self._render_3d_internal(ax, scene, frame_idx)
            return []

        return FuncAnimation(fig, update, frames=frames_range, interval=interval, blit=False)

    def save_animation(
        self,
        scene: SceneResult,
        output_path: str | Path,
        *,
        fps: float | None = None,
        figsize: tuple[float, float] | None = None,
        start_frame: int = 0,
        end_frame: int | None = None,
        dpi: int = 100,
        writer: str = "ffmpeg",
    ) -> None:
        """Save 3D animation as video file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        anim = self.create_animation(
            scene,
            fps=fps,
            figsize=figsize,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        anim.save(str(output_path), writer=writer, fps=fps or scene.fps, dpi=dpi)

    def _render_3d_internal(self, ax: Axes3D, scene: SceneResult, frame_idx: int) -> None:
        self.court_renderer.render_3d(ax, show_net=True)

        track_ids = self._get_player_tracks(scene)
        players_position = self._get_players_position(scene)
        players_yaw = self._get_players_yaw(scene)
        players_smpl = self._build_players_smpl_vertices_court(scene)

        if self.style.player_representation == "smpl":
            for player_idx, track_id in enumerate(track_ids):
                color = self._player_color(player_idx)
                vertices = players_smpl[player_idx, frame_idx]
                self._render_smpl_mesh_3d(
                    ax,
                    vertices,
                    color=color,
                    alpha=self.style.mesh_alpha,
                )
                ax.text(
                    players_position[player_idx, frame_idx, 0],
                    players_position[player_idx, frame_idx, 1],
                    players_position[player_idx, frame_idx, 2] + 1.0,
                    f"P{track_id}",
                    color=color,
                )
        else:
            players_kp_3d = self._get_players_kp_3d(scene)
            for player_idx, track_id in enumerate(track_ids):
                color = self._player_color(player_idx)
                style_override = SkeletonStyle(
                    joint_color=color,
                    bone_color=color,
                    joint_size=(self.style.skeleton_style.joint_size if self.style.skeleton_style else 5.0),
                    bone_width=(self.style.skeleton_style.bone_width if self.style.skeleton_style else 2.0),
                    joint_alpha=(self.style.skeleton_style.joint_alpha if self.style.skeleton_style else 1.0),
                    bone_alpha=(self.style.skeleton_style.bone_alpha if self.style.skeleton_style else 0.8),
                )
                self.skeleton_renderer.render_3d(
                    ax,
                    players_kp_3d[player_idx, frame_idx],
                    style_override=style_override,
                )
                ax.text(
                    players_position[player_idx, frame_idx, 0],
                    players_position[player_idx, frame_idx, 1],
                    players_position[player_idx, frame_idx, 2] + 1.0,
                    f"P{track_id}",
                    color=color,
                )

        if self.style.show_direction:
            for player_idx in range(players_position.shape[0]):
                pos = players_position[player_idx, frame_idx]
                yaw = players_yaw[player_idx, frame_idx]
                dx = np.sin(yaw)
                dy = np.cos(yaw)
                ax.quiver(pos[0], pos[1], pos[2] + 0.5, dx, dy, 0, color="yellow", arrow_length_ratio=0.3)

        if scene.ball_3d is not None:
            ball_pos = scene.ball_3d[frame_idx]
            if np.isfinite(ball_pos).all():
                self.ball_renderer.render_ball_3d(ax, ball_pos, label="Ball")

            if self.style.show_trail:
                start_idx = max(0, frame_idx - self.style.trail_length)
                trail = scene.ball_3d[start_idx : frame_idx + 1]
                valid = np.isfinite(trail).all(axis=-1)
                if valid.sum() > 1:
                    self.ball_renderer.render_trajectory_3d(
                        ax,
                        trail[valid],
                        show_start_end=False,
                    )

        center_x = float(np.mean(players_position[:, frame_idx, 0]))
        center_y = float(np.mean(players_position[:, frame_idx, 1]))
        self._set_zoomed_view(ax, center_x=center_x, center_y=center_y)
        ax.set_title(f"Frame: {frame_idx}/{scene.num_frames}")

    def _set_zoomed_view(self, ax: Axes3D, center_x: float, center_y: float) -> None:
        x_half_span = 8.0
        y_half_span = 10.0

        ax.set_xlim(center_x - x_half_span, center_x + x_half_span)
        ax.set_ylim(center_y - y_half_span, center_y + y_half_span)
        ax.set_zlim(0.0, 4.0)
        ax.set_box_aspect([x_half_span * 2.0, y_half_span * 2.0, 4.0])
