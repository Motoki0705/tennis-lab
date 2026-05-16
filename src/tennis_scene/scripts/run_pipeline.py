"""Run the tennis scene 3D reconstruction pipeline.

Usage:
    python -m src.tennis_scene.scripts.run_pipeline video_path=inputs/demo/match.mp4
    python -m src.tennis_scene.scripts.run_pipeline video_path=... max_frames=100

Notes:
        - The pipeline combines court keypoint detection, GVHMR, ball detection, PLCS,
            and BLCS.
    - Configuration is loaded from `src/tennis_scene/configs/pipeline.yaml`.
    - Hydra handles runtime overrides.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

LOGGER = logging.getLogger(__name__)


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="pipeline",
)
def main(cfg: DictConfig) -> int:
    """Run the tennis scene reconstruction pipeline."""
    from src.tennis_scene.pipeline import TennisSceneOrchestrator

    video_path = Path(to_absolute_path(str(cfg.video_path)))
    if not video_path.exists():
        LOGGER.error(f"Video not found: {video_path}")
        return 1

    output_dir = Path(to_absolute_path(str(cfg.output_dir)))
    output_name = cfg.get("output_name")
    if output_name is None:
        output_name = video_path.stem

    output_path = output_dir / f"{output_name}.npz"

    max_frames = cfg.get("max_frames")
    court_kp_frame = int(cfg.court_kp.frame_index)

    LOGGER.info("Configuration:")
    LOGGER.info(f"  Device: {cfg.device}")
    LOGGER.info(f"  Max frames: {max_frames}")
    LOGGER.info(f"  Court KP start frame: {court_kp_frame}")
    LOGGER.info(f"  Skip GVHMR: {cfg.gvhmr.get('skip', False)}")
    LOGGER.info(f"  Skip ball: {cfg.ball_detection.get('skip', False)}")
    LOGGER.info(f"  Skip BLCS: {cfg.blcs.get('skip', False)}")

    orchestrator = TennisSceneOrchestrator.from_config(cfg)

    LOGGER.info("Running pipeline...")
    result = orchestrator.run(
        video_path=video_path,
        max_frames=max_frames,
        court_kp_frame=court_kp_frame,
    )

    LOGGER.info("Saving results...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    LOGGER.info(f"Saved: {output_path}")

    LOGGER.info("=" * 60)
    LOGGER.info("Pipeline Summary:")
    LOGGER.info(f"  Frames processed: {result.num_frames}")
    LOGGER.info(f"  FPS: {result.fps:.2f}")
    LOGGER.info(f"  Resolution: {result.width}x{result.height}")
    if result.ball_3d is not None:
        visible_ball = result.ball_visibility.sum() if result.ball_visibility is not None else 0
        LOGGER.info(f"  Ball visible frames: {visible_ball}/{result.num_frames}")
    LOGGER.info("=" * 60)

    return 0


if __name__ == "__main__":
    main()
