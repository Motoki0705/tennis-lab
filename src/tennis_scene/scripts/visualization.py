"""Visualize tennis scene reconstruction results.

This script renders the output of the tennis scene pipeline as:
- Interactive matplotlib animation (view in UI)
- MP4 video file (saved to disk)

Example commands:
    `third_party/GVHMR/.venv/bin/python -m src.tennis_scene.scripts.visualization input=outputs/tennis_scene/clip.npz`
    `third_party/GVHMR/.venv/bin/python -m src.tennis_scene.scripts.visualization input=... view=2d output=animation.mp4`
    `third_party/GVHMR/.venv/bin/python -m src.tennis_scene.scripts.visualization input=... display=true`

Config entry point: `src/tennis_scene/configs/visualization.yaml`
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

LOGGER = logging.getLogger(__name__)


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

    # Create style
    style = TennisSceneStyle(
        player_color=str(cfg.style.player_color),
        trail_length=int(cfg.style.trail_length),
        show_direction=bool(cfg.style.show_direction),
        show_trail=bool(cfg.style.show_trail),
        figsize=tuple(cfg.style.figsize),
    )

    renderer = TennisSceneRenderer(style)

    # Determine frame range
    start_frame = int(cfg.get("start_frame", 0))
    end_frame = cfg.get("end_frame")
    if end_frame is not None:
        end_frame = int(end_frame)
    else:
        end_frame = scene.num_frames

    fps = cfg.get("fps")
    if fps is not None:
        fps = float(fps)
    else:
        fps = scene.fps

    view = str(cfg.view)
    display = bool(cfg.get("display", False))

    LOGGER.info(f"View: {view}, FPS: {fps:.1f}")
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
            view=view,
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
        anim = renderer.create_animation(
            scene,
            view=view,
            fps=fps,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        plt.show()

    # If neither save nor display, render single frame
    if output_path is None and not display:
        LOGGER.info("Rendering single frame preview")
        frame_idx = start_frame
        if view == "3d":
            fig, ax = renderer.render_frame_3d(scene, frame_idx)
        else:
            fig, ax = renderer.render_frame_2d(scene, frame_idx)
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
