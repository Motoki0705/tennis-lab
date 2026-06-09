"""Run the tennis scene 3D reconstruction pipeline.

Usage:
    python -m src.tennis_scene.scripts.run_pipeline video_paths='[inputs/demo/cam0.mp4,inputs/demo/cam1.mp4]'
    python -m src.tennis_scene.scripts.run_pipeline video_paths='[cam0.mp4,cam1.mp4]' max_frames=100

Notes:
    - The pipeline combines court keypoint detection, GVHMR, ball detection, PLCS,
      and BLCS.
    - Input videos must already be synchronized and share FPS, frame count, and resolution.
    - Configuration is loaded from `src/tennis_scene/configs/pipeline.yaml`.
    - Hydra handles runtime overrides.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, ListConfig

LOGGER = logging.getLogger(__name__)


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="pipeline",
)
def main(cfg: DictConfig) -> int:
    """Run the tennis scene reconstruction pipeline."""
    from src.tennis_scene.pipeline import TennisSceneOrchestrator

    raw_video_paths = cfg.get("video_paths")
    if not isinstance(raw_video_paths, (list, tuple, ListConfig)) or not raw_video_paths:
        LOGGER.error("video_paths must be a non-empty list")
        return 1
    video_paths = [Path(to_absolute_path(str(video_path))) for video_path in raw_video_paths]
    missing_paths = [video_path for video_path in video_paths if not video_path.exists()]
    if missing_paths:
        LOGGER.error(f"Video not found: {missing_paths[0]}")
        return 1

    raw_camera_ids = cfg.get("camera_ids")
    camera_ids = None
    if raw_camera_ids is not None:
        if not isinstance(raw_camera_ids, (list, tuple, ListConfig)):
            LOGGER.error("camera_ids must be null or a list")
            return 1
        camera_ids = [str(camera_id) for camera_id in raw_camera_ids]
        if len(camera_ids) != len(video_paths):
            LOGGER.error("camera_ids length must match video_paths length")
            return 1

    output_dir = Path(to_absolute_path(str(cfg.output_dir)))
    output_name = cfg.get("output_name")
    if output_name is None:
        output_name = video_paths[0].stem

    output_path = output_dir / f"{output_name}.npz"

    max_frames = cfg.get("max_frames")
    court_kp_annotation_frame = int(
        cfg.court_kp.get("annotation_frame_index", cfg.court_kp.get("frame_index", 0))
    )

    LOGGER.info("Configuration:")
    LOGGER.info(f"  Device: {cfg.device}")
    LOGGER.info(f"  Cameras: {len(video_paths)}")
    for index, video_path in enumerate(video_paths):
        camera_label = camera_ids[index] if camera_ids is not None else f"cam{index}"
        LOGGER.info(f"    {camera_label}: {video_path}")
    LOGGER.info(f"  Max frames: {max_frames}")
    LOGGER.info(f"  Court KP annotation frame: {court_kp_annotation_frame}")
    LOGGER.info(f"  Skip GVHMR: {cfg.gvhmr.get('skip', False)}")
    LOGGER.info(f"  Skip ball: {cfg.ball_detection.get('skip', False)}")
    LOGGER.info(f"  Skip BLCS: {cfg.blcs.get('skip', False)}")

    orchestrator = TennisSceneOrchestrator.from_config(cfg)

    LOGGER.info("Running pipeline...")
    result = orchestrator.run(
        video_paths=video_paths,
        max_frames=max_frames,
        court_kp_annotation_frame=court_kp_annotation_frame,
        camera_ids=camera_ids,
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
        visible_ball = 0
        if result.ball_visibility is not None:
            visible_ball = int(result.ball_visibility.any(axis=0).sum())
        LOGGER.info(f"  Ball visible frames: {visible_ball}/{result.num_frames}")
    LOGGER.info("=" * 60)

    return 0


if __name__ == "__main__":
    main()
