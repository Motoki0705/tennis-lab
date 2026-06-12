"""Visualize tennis scene reconstruction results.

Usage:
    python -m src.tennis_scene.scripts.visualization input=outputs/tennis_scene/clip.npz
    python -m src.tennis_scene.scripts.visualization input=... output=animation.mp4 style.player_representation=skeleton
    python -m src.tennis_scene.scripts.visualization input=... display=true

Notes:
    - The visualizer can render an interactive matplotlib 3D animation or save an MP4 file.
    - Configuration is loaded from `src/tennis_scene/configs/visualization.yaml`.
    - Hydra handles runtime overrides.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import matplotlib.pyplot as plt
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

if TYPE_CHECKING:
    from src.tennis_scene.io import SceneResult

LOGGER = logging.getLogger(__name__)

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
            "Visualization requires SMPL reconstruction fields: "
            f"{missing_str}."
        )


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="visualization",
)
def main(cfg: DictConfig) -> int:
    """Visualize tennis scene results."""
    from src.tennis_scene.io import SceneResult
    from src.tennis_scene.rendering import TennisSceneRenderer
    from src.tennis_scene.rendering.tennis_scene_renderer import TennisSceneStyle

    # Load input
    input_path = Path(to_absolute_path(str(cfg.input)))
    if not input_path.exists():
        LOGGER.error(f"Input file not found: {input_path}")
        return 1

    LOGGER.info(f"Loading scene from {input_path}")
    scene = SceneResult.load(input_path)

    LOGGER.info(f"Scene: {scene.num_frames} frames, {scene.fps:.1f} FPS")
    LOGGER.info(f"Resolution: {scene.width}x{scene.height}")
    _validate_scene_for_smpl(scene)

    # Create style
    style = TennisSceneStyle(
        trail_length=int(cfg.style.trail_length),
        show_direction=bool(cfg.style.show_direction),
        show_trail=bool(cfg.style.show_trail),
        figsize=tuple(cfg.style.figsize),
        player_representation=str(cfg.style.player_representation),
        mesh_alpha=float(cfg.style.mesh_alpha),
    )

    renderer = TennisSceneRenderer(style)

    # Determine frame range
    start_frame = int(cfg.get("start_frame", 0))
    end_frame = cfg.get("end_frame")
    end_frame = int(end_frame) if end_frame is not None else scene.num_frames

    fps = cfg.get("fps")
    fps = float(fps) if fps is not None else scene.fps

    display = bool(cfg.get("display", False))

    LOGGER.info(f"3D mode, FPS: {fps:.1f}")
    LOGGER.info(f"Frame range: {start_frame}-{end_frame}")

    # Output path
    output_path = cfg.get("output")
    if output_path is not None:
        output_path = Path(to_absolute_path(str(output_path)))

    # Create animation
    if output_path is not None:
        LOGGER.info(f"Saving animation to {output_path}")
        renderer.save_animation(
            scene,
            output_path,
            fps=fps,
            start_frame=start_frame,
            end_frame=end_frame,
            dpi=int(cfg.get("dpi", 100)),
            writer=str(cfg.get("writer", "ffmpeg")),
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
        plt.savefig("preview.png", dpi=150)
        LOGGER.info("Saved preview to preview.png")
        plt.close()

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main()
