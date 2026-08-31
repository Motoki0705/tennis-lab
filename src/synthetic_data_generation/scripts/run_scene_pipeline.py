"""Run the canonical mutable scene pipeline from one video and scene ID.

Usage:
    python -m src.synthetic_data_generation.scripts.run_scene_pipeline
    python -m src.synthetic_data_generation.scripts.run_scene_pipeline request.from_stage=alignment
    python -m src.synthetic_data_generation.scripts.run_scene_pipeline request.through_stage=alignment

Notes:
    - Hydra loads all runtime authority from `src/synthetic_data_generation/configs/run_scene_pipeline.yaml`.
    - The command publishes only fixed paths beneath the configured scene workspace.
"""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from src.synthetic_data_generation.configuration import (
    ScenePipelineConfiguration,
)
from src.synthetic_data_generation.pipeline.application import (
    build_scene_pipeline_runner,
)
from src.utils.hydra import hydra_main


@hydra_main(
    config_path="../configs",
    config_name="run_scene_pipeline",
    version_base="1.3",
    validation_boundary="synthetic.scene_pipeline",
)
def main(config: DictConfig) -> int:  # pragma: no cover - Hydra CLI boundary
    """Resolve, execute, and report the one canonical scene request."""
    runtime = ScenePipelineConfiguration.from_config(config)
    resolved_yaml = OmegaConf.to_yaml(config, resolve=True, sort_keys=True)
    runner = build_scene_pipeline_runner(
        runtime,
        resolved_config_yaml=resolved_yaml,
    )
    manifest = runner.run(runtime.request)
    terminal_stage = runtime.request.through_stage
    terminal_definition = runner.registry.definition(terminal_stage)
    print(f"scene={runtime.request.scene_id}")
    print(f"run_manifest={runtime.workspace.run_manifest_path}")
    print(f"terminal_stage={terminal_stage.value}")
    print(f"output={runtime.workspace.owner_path(terminal_definition)}")
    print(f"status={manifest.stages[terminal_stage].status.value}")
    return 0


if __name__ == "__main__":  # pragma: no cover - module execution
    main()
