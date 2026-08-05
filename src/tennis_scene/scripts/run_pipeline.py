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

from omegaconf import DictConfig

from src.tennis_scene.configuration import validate_pipeline_boundary
from src.utils.configuration import PathRole
from src.utils.hydra import hydra_main, register_boundary_validator

LOGGER = logging.getLogger(__name__)
_BOUNDARY = "tennis_scene.pipeline"
register_boundary_validator(_BOUNDARY, validate_pipeline_boundary)


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="pipeline",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> int:
    """Run the tennis scene reconstruction pipeline."""
    from src.tennis_scene.configuration import PipelineRuntimeConfig
    from src.tennis_scene.pipeline import TennisSceneOrchestrator

    runtime = PipelineRuntimeConfig.from_config(cfg)
    missing_paths = [path for path in runtime.video_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Video not found: {missing_paths[0]}")

    LOGGER.info("Configuration:")
    LOGGER.info(f"  Device: {runtime.device}")
    LOGGER.info(f"  Cameras: {len(runtime.video_paths)}")
    for camera_label, video_path in zip(
        runtime.camera_ids, runtime.video_paths, strict=True
    ):
        LOGGER.info(f"    {camera_label}: {video_path}")
    LOGGER.info(f"  Max frames: {runtime.max_frames}")
    LOGGER.info(f"  Court KP frame index: {runtime.frame_index}")

    orchestrator = TennisSceneOrchestrator.from_runtime_config(runtime)

    LOGGER.info("Running pipeline...")
    result = orchestrator.run(
        video_paths=runtime.video_paths,
        video_role=PathRole.DATA,
        max_frames=runtime.max_frames,
        frame_index=runtime.frame_index,
        camera_ids=runtime.camera_ids,
    )

    LOGGER.info("Saving results...")
    runtime.output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(runtime.output_path)
    LOGGER.info(f"Saved: {runtime.output_path}")

    LOGGER.info("=" * 60)
    LOGGER.info("Pipeline Summary:")
    LOGGER.info(f"  Frames processed: {result.num_frames}")
    LOGGER.info(f"  FPS: {result.fps:.2f}")
    LOGGER.info(f"  Resolution: {result.width}x{result.height}")
    if result.ball_3d is not None:
        visible_ball = 0
        if result.ball_vis is not None:
            visible_ball = int(result.ball_vis.any(axis=0).sum())
        LOGGER.info(f"  Ball visible frames: {visible_ball}/{result.num_frames}")
    LOGGER.info("=" * 60)

    return 0


if __name__ == "__main__":
    main()
