"""Run the configured reconstruction stages and save one SceneResult archive.

See src/tennis_scene/README.md for the canonical input/output layout, path
roots, required files, and executable examples.
"""

from __future__ import annotations

import logging

from omegaconf import DictConfig

from src.tennis_scene.configuration import validate_pipeline_boundary
from src.tennis_scene.schema import SceneResult
from src.utils.configuration import PathRole
from src.utils.hydra import hydra_main, register_boundary_validator

LOGGER = logging.getLogger(__name__)
_BOUNDARY = "tennis_scene.pipeline"
register_boundary_validator(_BOUNDARY, validate_pipeline_boundary)


def log_pipeline_summary(result: SceneResult) -> None:
    """Report observed 2D and valid 3D frames from their independent masks."""
    LOGGER.info("=" * 60)
    LOGGER.info("Pipeline Summary:")
    LOGGER.info(f"  Frames processed: {result.num_frames}")
    LOGGER.info(f"  FPS: {result.fps:.2f}")
    LOGGER.info(f"  Resolution: {result.width}x{result.height}")
    observed = "unavailable" if result.ball_vis is None else f"{int(result.ball_vis.any(axis=0).sum())}/{result.num_frames}"
    valid = "unavailable" if result.ball_3d_valid is None else f"{int(result.ball_3d_valid.sum())}/{result.num_frames}"
    LOGGER.info(f"  Ball 2D observed frames (>=1 view): {observed}")
    LOGGER.info(f"  Ball 3D valid frames: {valid}")
    LOGGER.info("=" * 60)


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="pipeline",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> int:
    """Run the tennis scene reconstruction pipeline."""
    from src.tennis_scene.archive import save_scene_result
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

    orchestrator = TennisSceneOrchestrator.from_runtime_config(runtime)

    LOGGER.info("Running pipeline...")
    result = orchestrator.run(
        video_paths=runtime.video_paths,
        video_role=PathRole.DATA,
        max_frames=runtime.max_frames,
        camera_ids=runtime.camera_ids,
        store_root=None,  # cache.directory/<source digest>
    )

    LOGGER.info("Saving results...")
    runtime.output_path.parent.mkdir(parents=True, exist_ok=True)
    save_scene_result(result, runtime.output_path)
    LOGGER.info(f"Saved: {runtime.output_path}")

    log_pipeline_summary(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
