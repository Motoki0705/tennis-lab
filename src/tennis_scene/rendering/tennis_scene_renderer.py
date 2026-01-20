"""Tennis scene renderer for complete scene visualization.

This module provides rendering of complete tennis scenes combining:
- Court rendering (lines, net)
- Ball trajectory rendering
- Player skeleton/mesh rendering (including SMPL mesh)

Supports both 2D and 3D views with animation capability.

Example:
    >>> from src.tennis_scene.rendering import TennisSceneRenderer
    >>> from src.tennis_scene.io import SceneResult
    >>>
    >>> result = SceneResult.load("output.npz")
    >>> renderer = TennisSceneRenderer()
    >>> anim = renderer.create_animation(result, view="3d")
    >>> plt.show()
    >>>
    >>> # Save as MP4
    >>> renderer.save_animation(result, "output.mp4", view="3d", fps=30)
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.utils.geometry.court import HALF_DOUBLES_WIDTH, HALF_LENGTH
from src.utils.rendering.ball_renderer import BallRenderer, BallStyle
from src.utils.rendering.court_renderer import CourtRenderer, CourtStyle
from src.utils.rendering.skeleton_renderer import SkeletonRenderer

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D
    from numpy.typing import NDArray

    from src.tennis_scene.io import SceneResult

# Default path to SMPL model for faces data
DEFAULT_SMPL_MODEL_PATH = Path("data/smplh/neutral/model.npz")


@dataclass
class TennisSceneStyle:
    """Style configuration for tennis scene rendering.

    Attributes:
        court_style: Court rendering style.
        ball_style: Ball rendering style.
        player_color: Color for player markers.
        mesh_color: Color for SMPL mesh (facecolor).
        mesh_alpha: Alpha for SMPL mesh.
        trail_length: Number of frames for movement trail.
        show_direction: Whether to show player facing direction.
        show_trail: Whether to show movement trail.
        show_mesh: Whether to render SMPL mesh (if available).
        figsize: Default figure size.

    """

    court_style: CourtStyle | None = None
    ball_style: BallStyle | None = None
    player_color: str = "#FF4444"
    mesh_color: str = "#7BC8F6"
    mesh_alpha: float = 0.6
    trail_length: int = 30
    show_direction: bool = True
    show_trail: bool = True
    show_mesh: bool = True
    figsize: tuple[float, float] = (12, 8)


class TennisSceneRenderer:
    """Render complete tennis scenes with court, ball, and player.

    Combines all scene elements for comprehensive visualization.
    Supports 2D top-down, 3D perspective, and animated views.
    Can render SMPL mesh when smpl_vertices_global is available.

    Example:
        >>> renderer = TennisSceneRenderer()
        >>> result = SceneResult.load("output.npz")
        >>>
        >>> # Static frame
        >>> fig, ax = renderer.render_frame_3d(result, frame_idx=0)
        >>>
        >>> # Animation
        >>> anim = renderer.create_animation(result, view="3d")
        >>> anim.save("output.mp4", fps=30)

    """

    def __init__(
        self,
        style: TennisSceneStyle | None = None,
        smpl_model_path: str | Path | None = None,
    ) -> None:
        """Initialize tennis scene renderer.

        Args:
            style: Style configuration. If None, uses defaults.
            smpl_model_path: Path to SMPL model file for mesh faces.
                If None, uses default path.

        """
        self.style = style or TennisSceneStyle()
        self.court_renderer = CourtRenderer(self.style.court_style)
        self.ball_renderer = BallRenderer(self.style.ball_style)
        self.skeleton_renderer = SkeletonRenderer(skeleton_type="coco17")

        # Load SMPL faces for mesh rendering
        self._smpl_faces: NDArray[np.int64] | None = None
        if self.style.show_mesh:
            model_path = smpl_model_path or DEFAULT_SMPL_MODEL_PATH
            self._smpl_faces = self._load_smpl_faces(model_path)

    def _load_smpl_faces(self, model_path: str | Path) -> NDArray[np.int64] | None:
        """Load SMPL faces from model file.

        Args:
            model_path: Path to SMPL model file (.npz or .pkl).

        Returns:
            SMPL faces array (F, 3) or None if loading fails.

        """
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"Warning: SMPL model not found at {model_path}, mesh rendering disabled")
            return None

        try:
            if model_path.suffix == ".npz":
                data = np.load(model_path)
                faces = data["f"].astype(np.int64)
            elif model_path.suffix == ".pkl":
                with open(model_path, "rb") as f:
                    data = pickle.load(f, encoding="latin1")
                faces = np.array(data["f"]).astype(np.int64)
            else:
                print(f"Warning: Unknown SMPL model format: {model_path.suffix}")
                return None
            return faces
        except Exception as e:
            print(f"Warning: Failed to load SMPL faces: {e}")
            return None

    def _render_smpl_mesh_3d(
        self,
        ax: Axes3D,
        vertices: NDArray[np.float32],
        color: str | None = None,
        alpha: float | None = None,
    ) -> None:
        """Render SMPL mesh in 3D axes.

        Args:
            ax: 3D matplotlib axes.
            vertices: SMPL vertices (V, 3).
            color: Mesh face color (uses style default if None).
            alpha: Mesh alpha (uses style default if None).

        """
        if self._smpl_faces is None:
            return

        color = color or self.style.mesh_color
        alpha = alpha if alpha is not None else self.style.mesh_alpha

        # Create triangles from faces
        triangles = vertices[self._smpl_faces]

        # Create Poly3DCollection
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
        """Render a single frame in 3D.

        Args:
            scene: Scene result data.
            frame_idx: Frame index to render.
            figsize: Figure size (uses default if None).
            title: Custom title.
            ax: Existing 3D axes to draw on (creates new if None).

        Returns:
            Tuple of (figure, 3D axes). Figure is None if ax was provided.

        """
        fig = None
        if ax is None:
            figsize = figsize or self.style.figsize
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")

        # Render court
        self.court_renderer.render_3d(ax, show_net=True)

        # Render SMPL mesh if available
        if (
            self.style.show_mesh
            and scene.smpl_vertices_global is not None
            and self._smpl_faces is not None
        ):
            vertices = scene.smpl_vertices_global[frame_idx]
            self._render_smpl_mesh_3d(ax, vertices)

        # Render player position (as fallback or additional indicator)
        pos = scene.player_position[frame_idx]
        if not self.style.show_mesh or scene.smpl_vertices_global is None:
            ax.scatter([pos[0]], [pos[1]], [pos[2] if len(pos) > 2 else 0],
                       c=self.style.player_color, s=100, marker="o", label="Player")

        # Render player direction
        if self.style.show_direction:
            yaw = scene.player_yaw[frame_idx]
            arrow_length = 1.0
            dx = arrow_length * np.sin(yaw)
            dy = arrow_length * np.cos(yaw)
            ax.quiver(
                pos[0], pos[1], (pos[2] if len(pos) > 2 else 0) + 0.5,
                dx, dy, 0,
                color="yellow", arrow_length_ratio=0.3, linewidth=2,
            )

        # Render ball if available
        if scene.ball_3d is not None:
            ball_pos = scene.ball_3d[frame_idx]
            if np.isfinite(ball_pos).all():
                ax.scatter([ball_pos[0]], [ball_pos[1]], [ball_pos[2]],
                           c="#CCFF00", s=80, marker="o", label="Ball")

        # Render ball trail
        if scene.ball_3d is not None and self.style.show_trail:
            start_idx = max(0, frame_idx - self.style.trail_length)
            trail = scene.ball_3d[start_idx:frame_idx + 1]
            valid = np.isfinite(trail).all(axis=-1)
            if valid.sum() > 1:
                valid_trail = trail[valid]
                ax.plot(valid_trail[:, 0], valid_trail[:, 1], valid_trail[:, 2],
                        c="#CCFF00", alpha=0.5, linewidth=2)

        if title is None:
            title = f"Frame: {frame_idx}/{scene.num_frames}"
        ax.set_title(title)

        return fig, ax

    def render_frame_2d(
        self,
        scene: SceneResult,
        frame_idx: int,
        *,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        ax: Axes | None = None,
    ) -> tuple[Figure | None, Axes]:
        """Render a single frame in 2D top-down view.

        Args:
            scene: Scene result data.
            frame_idx: Frame index to render.
            figsize: Figure size (uses default if None).
            title: Custom title.
            ax: Existing axes to draw on (creates new if None).

        Returns:
            Tuple of (figure, axes). Figure is None if ax was provided.

        """
        fig = None
        if ax is None:
            figsize = figsize or self.style.figsize
            fig, ax = plt.subplots(figsize=figsize)

        # Render court
        self.court_renderer.render_2d(ax, show_fence=True)

        # Render player position
        pos = scene.player_position[frame_idx]
        ax.scatter([pos[0]], [pos[1]], c=self.style.player_color, s=100,
                   zorder=5, label="Player")

        # Render player trail
        if self.style.show_trail:
            start_idx = max(0, frame_idx - self.style.trail_length)
            trail = scene.player_position[start_idx:frame_idx + 1]
            ax.plot(trail[:, 0], trail[:, 1], c="cyan", linewidth=2,
                    alpha=0.5, zorder=4)

        # Render player direction
        if self.style.show_direction:
            yaw = scene.player_yaw[frame_idx]
            arrow_length = 1.0
            dx = arrow_length * np.sin(yaw)
            dy = arrow_length * np.cos(yaw)
            ax.arrow(pos[0], pos[1], dx, dy, head_width=0.3, head_length=0.2,
                     fc="yellow", ec="black", zorder=6)

        # Render ball if available
        if scene.ball_3d is not None:
            ball_pos = scene.ball_3d[frame_idx]
            if np.isfinite(ball_pos).all():
                ax.scatter([ball_pos[0]], [ball_pos[1]], c="#CCFF00", s=80,
                           zorder=5, label="Ball", edgecolors="black", linewidths=1)

        # Render ball trail (2D projection)
        if scene.ball_3d is not None and self.style.show_trail:
            start_idx = max(0, frame_idx - self.style.trail_length)
            trail = scene.ball_3d[start_idx:frame_idx + 1]
            valid = np.isfinite(trail).all(axis=-1)
            if valid.sum() > 1:
                valid_trail = trail[valid]
                ax.plot(valid_trail[:, 0], valid_trail[:, 1],
                        c="#CCFF00", alpha=0.5, linewidth=2, zorder=4)

        if title is None:
            title = f"Frame: {frame_idx}/{scene.num_frames}"
        ax.set_title(title)

        return fig, ax

    def create_animation(
        self,
        scene: SceneResult,
        view: Literal["3d", "2d"] = "3d",
        *,
        fps: float | None = None,
        figsize: tuple[float, float] | None = None,
        start_frame: int = 0,
        end_frame: int | None = None,
    ) -> FuncAnimation:
        """Create animation of scene.

        Args:
            scene: Scene result data.
            view: View type ("3d" or "2d").
            fps: Frames per second. If None, uses scene FPS.
            figsize: Figure size.
            start_frame: Starting frame index.
            end_frame: Ending frame index (None for all frames).

        Returns:
            FuncAnimation object.

        """
        if fps is None:
            fps = scene.fps

        if end_frame is None:
            end_frame = scene.num_frames

        interval = 1000.0 / fps
        frames_range = range(start_frame, end_frame)

        figsize = figsize or self.style.figsize

        if view == "3d":
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")

            def update_3d(frame_idx: int) -> list:
                ax.clear()
                self._render_3d_internal(ax, scene, frame_idx)
                return []

            return FuncAnimation(
                fig, update_3d, frames=frames_range, interval=interval, blit=False
            )

        else:  # 2d
            fig, ax = plt.subplots(figsize=figsize)

            def update_2d(frame_idx: int) -> list:
                ax.clear()
                self._render_2d_internal(ax, scene, frame_idx)
                return []

            return FuncAnimation(
                fig, update_2d, frames=frames_range, interval=interval, blit=False
            )

    def save_animation(
        self,
        scene: SceneResult,
        output_path: str | Path,
        view: Literal["3d", "2d"] = "3d",
        *,
        fps: float | None = None,
        figsize: tuple[float, float] | None = None,
        start_frame: int = 0,
        end_frame: int | None = None,
        dpi: int = 100,
        writer: str = "ffmpeg",
    ) -> None:
        """Save animation as video file.

        Args:
            scene: Scene result data.
            output_path: Output video path (e.g., "output.mp4").
            view: View type ("3d" or "2d").
            fps: Frames per second. If None, uses scene FPS.
            figsize: Figure size.
            start_frame: Starting frame index.
            end_frame: Ending frame index (None for all frames).
            dpi: DPI for output.
            writer: Animation writer (e.g., "ffmpeg", "pillow").

        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if fps is None:
            fps = scene.fps

        anim = self.create_animation(
            scene, view=view, fps=fps, figsize=figsize,
            start_frame=start_frame, end_frame=end_frame
        )

        anim.save(str(output_path), writer=writer, fps=fps, dpi=dpi)
        print(f"Saved animation to {output_path}")

    def _render_3d_internal(
        self, ax: Axes3D, scene: SceneResult, frame_idx: int
    ) -> None:
        """Internal 3D rendering for animation updates."""
        self.court_renderer.render_3d(ax, show_net=True)

        # Render SMPL mesh if available
        if (
            self.style.show_mesh
            and scene.smpl_vertices_global is not None
            and self._smpl_faces is not None
        ):
            vertices = scene.smpl_vertices_global[frame_idx]
            self._render_smpl_mesh_3d(ax, vertices)

        # Player position marker (fallback or additional indicator)
        pos = scene.player_position[frame_idx]
        if not self.style.show_mesh or scene.smpl_vertices_global is None:
            # Only show marker if mesh is not rendered
            ax.scatter([pos[0]], [pos[1]], [pos[2] if len(pos) > 2 else 0],
                       c=self.style.player_color, s=100, marker="o")

        # Direction
        if self.style.show_direction:
            yaw = scene.player_yaw[frame_idx]
            dx = np.sin(yaw)
            dy = np.cos(yaw)
            ax.quiver(pos[0], pos[1], (pos[2] if len(pos) > 2 else 0) + 0.5,
                      dx, dy, 0, color="yellow", arrow_length_ratio=0.3)

        # Ball
        if scene.ball_3d is not None:
            ball_pos = scene.ball_3d[frame_idx]
            if np.isfinite(ball_pos).all():
                ax.scatter([ball_pos[0]], [ball_pos[1]], [ball_pos[2]],
                           c="#CCFF00", s=80, marker="o")

            # Trail
            if self.style.show_trail:
                start_idx = max(0, frame_idx - self.style.trail_length)
                trail = scene.ball_3d[start_idx:frame_idx + 1]
                valid = np.isfinite(trail).all(axis=-1)
                if valid.sum() > 1:
                    valid_trail = trail[valid]
                    ax.plot(valid_trail[:, 0], valid_trail[:, 1], valid_trail[:, 2],
                            c="#CCFF00", alpha=0.5, linewidth=2)

        ax.set_title(f"Frame: {frame_idx}/{scene.num_frames}")

    def _render_2d_internal(
        self, ax: Axes, scene: SceneResult, frame_idx: int
    ) -> None:
        """Internal 2D rendering for animation updates."""
        self.court_renderer.render_2d(ax, show_fence=True)

        # Player
        pos = scene.player_position[frame_idx]
        ax.scatter([pos[0]], [pos[1]], c=self.style.player_color, s=100, zorder=5)

        # Trail
        if self.style.show_trail:
            start_idx = max(0, frame_idx - self.style.trail_length)
            trail = scene.player_position[start_idx:frame_idx + 1]
            ax.plot(trail[:, 0], trail[:, 1], c="cyan", linewidth=2, alpha=0.5, zorder=4)

        # Direction
        if self.style.show_direction:
            yaw = scene.player_yaw[frame_idx]
            dx = np.sin(yaw)
            dy = np.cos(yaw)
            ax.arrow(pos[0], pos[1], dx, dy, head_width=0.3, head_length=0.2,
                     fc="yellow", ec="black", zorder=6)

        # Ball
        if scene.ball_3d is not None:
            ball_pos = scene.ball_3d[frame_idx]
            if np.isfinite(ball_pos).all():
                ax.scatter([ball_pos[0]], [ball_pos[1]], c="#CCFF00", s=80,
                           zorder=5, edgecolors="black", linewidths=1)

            # Trail
            if self.style.show_trail:
                start_idx = max(0, frame_idx - self.style.trail_length)
                trail = scene.ball_3d[start_idx:frame_idx + 1]
                valid = np.isfinite(trail).all(axis=-1)
                if valid.sum() > 1:
                    valid_trail = trail[valid]
                    ax.plot(valid_trail[:, 0], valid_trail[:, 1],
                            c="#CCFF00", alpha=0.5, linewidth=2, zorder=4)

        ax.set_title(f"Frame: {frame_idx}/{scene.num_frames}")


if __name__ == "__main__":
    # Smoke test
    print("TennisSceneRenderer: Complete tennis scene visualization")
    print("Use TennisSceneRenderer() to create")

    # Test style creation
    style = TennisSceneStyle(
        player_color="red",
        trail_length=20,
        show_direction=True,
        show_mesh=True,
    )
    renderer = TennisSceneRenderer(style)

    # Create dummy scene for testing
    from src.tennis_scene.io import SceneResult

    # Create dummy SMPL vertices (T, V, 3) - SMPL has 6890 vertices
    num_vertices = 6890
    dummy_vertices = np.random.rand(10, num_vertices, 3).astype(np.float32) * 2

    dummy_scene = SceneResult(
        num_frames=10,
        fps=30.0,
        width=1920,
        height=1080,
        court_kp=np.random.rand(20, 2).astype(np.float32),
        court_vis=np.ones(20, dtype=np.float32),
        player_position=np.random.rand(10, 3).astype(np.float32) * 5,
        player_yaw=np.random.rand(10).astype(np.float32) * np.pi,
        smpl_body_pose=np.random.rand(10, 63).astype(np.float32),
        smpl_global_orient=np.random.rand(10, 3).astype(np.float32),
        smpl_betas=np.random.rand(10).astype(np.float32),
        smpl_vertices_global=dummy_vertices,
        ball_3d=np.random.rand(10, 3).astype(np.float32) * 5,
    )

    # Test 2D rendering (no display)
    fig, ax = renderer.render_frame_2d(dummy_scene, frame_idx=0)
    plt.close(fig)
    print("2D rendering test passed!")

    # Test 3D rendering (no display)
    fig, ax = renderer.render_frame_3d(dummy_scene, frame_idx=0)
    plt.close(fig)
    print("3D rendering test passed!")

    # Test mesh loading
    if renderer._smpl_faces is not None:
        print(f"SMPL faces loaded: {renderer._smpl_faces.shape}")
    else:
        print("SMPL faces not loaded (model file may not exist)")

    print("Smoke test passed!")
