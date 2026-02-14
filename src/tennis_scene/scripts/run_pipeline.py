"""Run tennis scene 3D reconstruction pipeline.

This script runs the integrated pipeline combining:
- Court KP Detection (single frame, fixed camera)
- GVHMR (local SMPL, static_cam=True)
- WASB (ball detection)
- Trajectory Completion (optional)
- UV Event Detection
- PLCS (3D player position + yaw)
- BLCS (3D ball trajectory)
- 3D Event Detection

Example commands:
    `uv run python -m src.tennis_scene.scripts.run_pipeline video_path=inputs/demo/match.mp4`
    `uv run python -m src.tennis_scene.scripts.run_pipeline video_path=... max_frames=100`

Config entry point: `src/tennis_scene/configs/pipeline.yaml`
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
    LOGGER.info(f"  Court KP frame: {court_kp_frame}")
    LOGGER.info(f"  Skip GVHMR: {cfg.gvhmr.get('skip', False)}")
    LOGGER.info(f"  Skip ball: {cfg.wasb.get('skip', False)}")
    LOGGER.info(f"  Skip trajectory: {cfg.trajectory.get('skip', True)}")
    LOGGER.info(f"  Skip UV event: {cfg.event_uv.get('skip', True)}")
    LOGGER.info(f"  Skip BLCS: {cfg.blcs.get('skip', False)}")
    LOGGER.info(f"  Skip 3D event: {cfg.event_3d.get('skip', True)}")

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
    if result.event_uv_peak_mask is not None and result.event_uv_names is not None:
        uv_counts = result.event_uv_peak_mask.sum(axis=0).tolist()
        LOGGER.info(f"  UV events ({result.event_uv_names}): {uv_counts}")
    if result.event_3d_peak_mask is not None and result.event_3d_names is not None:
        event_3d_counts = result.event_3d_peak_mask.sum(axis=0).tolist()
        LOGGER.info(f"  3D events ({result.event_3d_names}): {event_3d_counts}")
    LOGGER.info("=" * 60)

    return 0


if __name__ == "__main__":
    main()
