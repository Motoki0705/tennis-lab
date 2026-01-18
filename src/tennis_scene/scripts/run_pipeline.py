"""Run tennis scene 3D reconstruction pipeline.

This script runs the integrated pipeline combining:
- Court KP Detection (single frame, fixed camera)
- GVHMR (local SMPL, static_cam=True)
- WASB (ball detection)
- PLCS (3D player position + yaw)
- BLCS (3D ball trajectory)

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


def _resolve_checkpoint(path: str | None) -> Path | None:
    """Resolve checkpoint path to absolute path."""
    if path is None:
        return None
    return Path(to_absolute_path(str(path)))


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="pipeline",
)
def main(cfg: DictConfig) -> int:
    """Run the tennis scene reconstruction pipeline."""
    from src.tennis_scene.pipeline import TennisScenePipeline

    LOGGER.info("=" * 60)
    LOGGER.info("Tennis Scene 3D Reconstruction Pipeline")
    LOGGER.info("=" * 60)

    video_path = Path(to_absolute_path(str(cfg.video_path)))
    if not video_path.exists():
        LOGGER.error(f"Video not found: {video_path}")
        return 1

    LOGGER.info(f"Video: {video_path}")

    output_dir = Path(to_absolute_path(str(cfg.output_dir)))
    output_name = cfg.get("output_name")
    if output_name is None:
        output_name = video_path.stem

    output_path = output_dir / f"{output_name}.npz"

    court_kp_ckpt = _resolve_checkpoint(cfg.court_kp.checkpoint)
    wasb_ckpt = _resolve_checkpoint(cfg.wasb.checkpoint)
    plcs_ckpt = _resolve_checkpoint(cfg.plcs.checkpoint)
    blcs_ckpt = _resolve_checkpoint(cfg.blcs.checkpoint)
    gvhmr_ckpt = _resolve_checkpoint(cfg.gvhmr.checkpoint)

    wasb_completion_ckpt = None
    if cfg.wasb.completion.enabled:
        wasb_completion_ckpt = _resolve_checkpoint(cfg.wasb.completion.checkpoint)

    device = str(cfg.device)
    max_frames = cfg.get("max_frames")
    court_kp_frame = int(cfg.court_kp.frame_index)
    skip_ball = bool(cfg.wasb.get("skip", False))
    skip_gvhmr = bool(cfg.gvhmr.get("skip", False))

    LOGGER.info("Configuration:")
    LOGGER.info(f"  Device: {device}")
    LOGGER.info(f"  Max frames: {max_frames}")
    LOGGER.info(f"  Court KP frame: {court_kp_frame}")
    LOGGER.info(f"  Skip ball: {skip_ball}")
    LOGGER.info(f"  Skip GVHMR: {skip_gvhmr}")

    pipeline = TennisScenePipeline.from_checkpoints(
        court_kp_checkpoint=court_kp_ckpt,
        wasb_checkpoint=wasb_ckpt,
        plcs_checkpoint=plcs_ckpt,
        blcs_checkpoint=blcs_ckpt,
        device=device,
        wasb_batch_size=int(cfg.wasb.batch_size),
        wasb_completion_enabled=bool(cfg.wasb.completion.enabled),
        wasb_completion_checkpoint=wasb_completion_ckpt,
    )

    LOGGER.info("Running pipeline...")
    result = pipeline.run(
        video_path=video_path,
        gvhmr_checkpoint=gvhmr_ckpt,
        max_frames=max_frames,
        court_kp_frame=court_kp_frame,
        skip_ball=skip_ball,
        skip_gvhmr=skip_gvhmr,
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
