"""
Generate incremental tennis-scene pseudo annotations for exported clips.

Usage:
    python -m src.tennis_scene.scripts.generate_dataset dataset_directory=tennis_scene/dataset

Notes:
    - The generation boundary and the nested pipeline are both strictly validated.
    - A failed clip records a failure marker and makes the process return non-zero.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from hydra import compose
from omegaconf import DictConfig, OmegaConf

from src.tennis_scene.configuration import validate_generate_dataset_boundary
from src.utils.configuration import PathRole
from src.utils.hydra import hydra_main, register_boundary_validator

LOGGER = logging.getLogger(__name__)
_BOUNDARY = "tennis_scene.generate_dataset"
register_boundary_validator(_BOUNDARY, validate_generate_dataset_boundary)

if TYPE_CHECKING:
    from src.tennis_scene.configuration import (
        GenerateDatasetRuntimeConfig,
        PipelineRuntimeConfig,
    )


def _compose_pipeline_config(
    runtime: GenerateDatasetRuntimeConfig,
) -> tuple[DictConfig, PipelineRuntimeConfig]:
    """Compose the canonical pipeline config and validate explicit overrides."""
    from src.tennis_scene.configuration import PipelineRuntimeConfig

    pipeline_cfg = compose(
        config_name="pipeline",
        overrides=list(runtime.pipeline_overrides),
    )
    paths = {
        "project_root": str(runtime.roots.project_root),
        "data_root": str(runtime.roots.data_root),
        "checkpoint_root": str(runtime.roots.checkpoint_root),
        "artifact_root": str(runtime.roots.artifact_root),
        "output_root": str(runtime.roots.output_root),
        "cache_root": str(runtime.roots.cache_root),
        "external_asset_root": str(runtime.roots.external_asset_root),
    }
    merged_pipeline_cfg = OmegaConf.merge(pipeline_cfg, {"paths": paths})
    if not isinstance(merged_pipeline_cfg, DictConfig):
        raise TypeError("pipeline config must compose to a mapping")
    pipeline_cfg = merged_pipeline_cfg
    return pipeline_cfg, PipelineRuntimeConfig.from_config(pipeline_cfg)


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="generate_dataset",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> int:
    """Run pseudo annotation generation for selected or pending clips."""
    from src.tennis_scene.configuration import parse_generate_dataset_config
    from src.tennis_scene.generate_dataset import generate_pseudo_annotations
    from src.tennis_scene.io import SceneResult
    from src.tennis_scene.pipeline import TennisSceneOrchestrator

    runtime = parse_generate_dataset_config(cfg)
    pipeline_cfg, pipeline_runtime = _compose_pipeline_config(runtime)
    pipeline_yaml = OmegaConf.to_yaml(pipeline_cfg, resolve=True)
    orchestrator = TennisSceneOrchestrator.from_runtime_config(pipeline_runtime)

    def run_clip(video_paths: Sequence[Path], camera_ids: Sequence[str]) -> SceneResult:
        result: SceneResult = orchestrator.run(
            video_paths=video_paths,
            video_role=PathRole.ARTIFACT,
            camera_ids=camera_ids,
            max_frames=pipeline_runtime.max_frames,
            frame_index=pipeline_runtime.frame_index,
        )
        return result

    outcomes = generate_pseudo_annotations(
        runtime.dataset_directory,
        run_clip,
        pipeline_config_yaml=pipeline_yaml,
        clip_ids=runtime.clip_ids,
        overwrite=runtime.overwrite,
        continue_on_error=runtime.continue_on_error,
    )
    for outcome in outcomes:
        if outcome.status == "failed":
            LOGGER.error(f"{outcome.clip_id}: {outcome.error}")
        else:
            LOGGER.info(f"{outcome.clip_id}: {outcome.status}")
    failed = sum(outcome.status == "failed" for outcome in outcomes)
    return 1 if failed else 0


if __name__ == "__main__":
    main()
