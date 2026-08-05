"""Visualize tennis scene reconstruction results.

Usage:
    python -m src.tennis_scene.scripts.visualization input=tennis_scene/clip.npz
    python -m src.tennis_scene.scripts.visualization input=tennis_scene/clip.npz output=tennis_scene/animation.mp4

Notes:
    - The visualizer can render an interactive matplotlib 3D animation or save an MP4 file.
    - Configuration is loaded from `src/tennis_scene/configs/visualization.yaml`.
    - Input/output fragments are resolved under their declared artifact/output roots.
    - `camera` selects viewpoint and motion (presets: broadcast/side/top/corner/behind_far;
      modes: static/orbit/keyframes). `style.theme=dark` gives a broadcast-style look with
      HUD (ball speed, bounce count) and a top-down minimap.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from omegaconf import DictConfig

from src.tennis_scene.configuration import validate_visualization_boundary
from src.utils.hydra import hydra_main, register_boundary_validator

if TYPE_CHECKING:
    from src.tennis_scene.io import SceneResult

LOGGER = logging.getLogger(__name__)
_BOUNDARY = "tennis_scene.visualization"
register_boundary_validator(_BOUNDARY, validate_visualization_boundary)

_REQUIRED_SMPL_FIELDS = (
    "smpl_vertices_local",
    "smpl_global_orient",
    "player_position",
    "player_yaw",
)


def _validate_scene_for_smpl(scene: SceneResult) -> None:
    missing: list[str] = []
    for field in _REQUIRED_SMPL_FIELDS:
        if getattr(scene, field) is None:
            missing.append(field)
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            f"Visualization requires SMPL reconstruction fields: {missing_str}."
        )


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="visualization",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> int:
    """Visualize tennis scene results."""
    from src.tennis_scene.configuration import parse_visualization_config
    from src.tennis_scene.io import SceneResult
    from src.tennis_scene.rendering import TennisSceneRenderer
    from src.tennis_scene.rendering.tennis_scene_renderer import TennisSceneStyle

    runtime = parse_visualization_config(cfg)
    if not runtime.input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {runtime.input_path}")

    LOGGER.info(f"Loading scene from {runtime.input_path}")
    scene = SceneResult.load(runtime.input_path)

    LOGGER.info(f"Scene: {scene.num_frames} frames, {scene.fps:.1f} FPS")
    LOGGER.info(f"Resolution: {scene.width}x{scene.height}")
    _validate_scene_for_smpl(scene)

    style = TennisSceneStyle(
        trail_length=runtime.style.trail_length,
        show_trail=runtime.style.show_trail,
        figsize=runtime.style.figsize,
        player_representation=runtime.style.player_representation,
        mesh_alpha=runtime.style.mesh_alpha,
        theme=runtime.style.theme,
        show_ball_shadow=runtime.style.show_ball_shadow,
        show_player_shadow=runtime.style.show_player_shadow,
        show_player_trail=runtime.style.show_player_trail,
        player_trail_length=runtime.style.player_trail_length,
        show_bounces=runtime.style.show_bounces,
        show_hud=runtime.style.show_hud,
        show_minimap=runtime.style.show_minimap,
    )

    camera = runtime.camera
    LOGGER.info(
        f"Camera: mode={camera.mode}, base=({camera.base.elev:.0f}, "
        f"{camera.base.azim:.0f}, zoom={camera.base.zoom:.2f})"
    )

    renderer = TennisSceneRenderer(
        style,
        smpl_faces_path=runtime.smpl_faces_path,
        smpl_joint_regressor_path=runtime.smpl_joint_regressor_path,
        camera=camera,
    )

    # Determine frame range
    start_frame = runtime.start_frame
    end_frame = runtime.end_frame if runtime.end_frame is not None else scene.num_frames
    fps = runtime.fps if runtime.fps is not None else scene.fps
    display = runtime.display

    LOGGER.info(f"3D mode, FPS: {fps:.1f}")
    LOGGER.info(f"Frame range: {start_frame}-{end_frame}")

    # Output path
    output_path = runtime.output_path

    # Create animation
    if output_path is not None:
        LOGGER.info(f"Saving animation to {output_path}")
        renderer.save_animation(
            scene,
            output_path,
            fps=fps,
            start_frame=start_frame,
            end_frame=end_frame,
            dpi=runtime.dpi,
            writer=runtime.writer,
        )
        LOGGER.info("Animation saved successfully")

    # Display in UI
    if display:
        LOGGER.info("Displaying animation in UI (close window to exit)")
        renderer.create_animation(
            scene,
            fps=fps,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        plt.show()

    # If neither save nor display, render single frame
    if output_path is None and not display:
        LOGGER.info("Rendering single frame preview")
        frame_idx = start_frame
        fig, ax = renderer.render_frame_3d(scene, frame_idx)
        runtime.preview_output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(runtime.preview_output, dpi=150)
        LOGGER.info(f"Saved preview to {runtime.preview_output}")
        plt.close()

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main()
