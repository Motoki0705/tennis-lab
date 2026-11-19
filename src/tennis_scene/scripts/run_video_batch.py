"""Reconstruct one registered video using a single multi-clip execution plan."""

from __future__ import annotations

import logging

from omegaconf import DictConfig, OmegaConf

from src.tennis_scene.configuration import validate_video_batch_boundary
from src.utils.hydra import hydra_main, register_boundary_validator

LOGGER = logging.getLogger(__name__)
_BOUNDARY = "tennis_scene.run_video_batch"
register_boundary_validator(_BOUNDARY, validate_video_batch_boundary)


@hydra_main(version_base="1.3", config_path="../configs", config_name="run_video_batch", validation_boundary=_BOUNDARY)
def main(cfg: DictConfig) -> int:
    from src.tennis_scene.configuration import parse_video_batch_config
    from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
    from src.tennis_scene.pipeline.video_batch import run_video_batch
    from src.tennis_scene.scripts.generate_dataset import _compose_pipeline_config

    runtime = parse_video_batch_config(cfg)
    generation = runtime.generation
    pipeline_cfg, pipeline = _compose_pipeline_config(generation, config_name=runtime.pipeline_config)
    result = run_video_batch(generation.dataset_directory, runtime.video_id, TennisSceneOrchestrator.from_runtime_config(pipeline),
        pipeline_config_yaml=OmegaConf.to_yaml(pipeline_cfg, resolve=True), clip_ids=generation.clip_ids,
        overwrite=generation.overwrite, continue_on_error=generation.continue_on_error,
        progress_directory=runtime.progress_directory, batching=runtime.batching)
    for outcome in result.outcomes:
        LOGGER.log(logging.ERROR if outcome.status in {"failed", "blocked"} else logging.INFO,
                   "%s %s%s", outcome.clip_id, outcome.status, "" if outcome.error is None else f" {outcome.error}")
    LOGGER.info("Video progress %s", result.progress_path)
    if result.exit_code:
        raise SystemExit(result.exit_code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
