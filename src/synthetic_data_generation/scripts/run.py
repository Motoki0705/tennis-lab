"""Run video → NHT reconstruction → alignment → domain datasets.

Usage:
    python -m src.synthetic_data_generation.scripts.run scene_id=B00 input_video=source/tennis.mp4
    python -m src.synthetic_data_generation.scripts.run scene_id=B00 from_stage=alignment

Notes:
    - Hydra loads request configuration from ``src/synthetic_data_generation/configs/run.yaml``.
    - ``input_video`` is relative to the pipeline config's ``external_asset_root``.
    - ``pipeline_config`` is relative to the repository project root.
"""

from __future__ import annotations

import json

from omegaconf import DictConfig

from src.synthetic_data_generation.pipeline import PipelineRequest, run_scene_pipeline
from src.synthetic_data_generation.pipeline.config import (
    ScenePipelineRunConfig,
    validate_run_boundary,
)
from src.synthetic_data_generation.pipeline.stages import Stage, Target
from src.utils.hydra import hydra_main, register_boundary_validator
from src.utils.paths import PROJECT_ROOT

_BOUNDARY = "synthetic.scene_pipeline"
register_boundary_validator(_BOUNDARY, validate_run_boundary)


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="run",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> None:
    """Execute one strict scene-pipeline request."""
    runtime = ScenePipelineRunConfig.from_config(cfg)
    run_path = run_scene_pipeline(
        PipelineRequest(
            scene_id=runtime.scene_id,
            config_path=runtime.pipeline_config,
            repository_root=PROJECT_ROOT.resolve(),
            input_video=runtime.input_video,
            from_stage=Stage(runtime.from_stage),
            targets=tuple(Target(value) for value in runtime.targets),
            nht_from_stage=runtime.nht_from_stage,
        )
    )
    print(json.dumps(json.loads(run_path.read_text()), indent=2))


if __name__ == "__main__":
    main()
