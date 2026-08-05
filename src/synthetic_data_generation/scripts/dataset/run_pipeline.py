"""
Plan or execute the generic path-driven synthetic-data pipeline.

Usage:
    python -m src.synthetic_data_generation.scripts.dataset.run_pipeline
    python -m src.synthetic_data_generation.scripts.dataset.run_pipeline execute=true

Notes:
    - Hydra loads configuration from `src/synthetic_data_generation/configs/dataset`.
    - `renderer.mode=prepared_outputs` explicitly selects prepared-output copies.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from src.synthetic_data_generation.configuration import validate_config
from src.synthetic_data_generation.dataset.execution import (
    execute_pipeline,
    validate_pipeline_inputs,
)
from src.synthetic_data_generation.dataset.pipeline import PathPipelineManifest
from src.utils.configuration import PathRole
from src.utils.hydra import hydra_main


def _mapping(value: DictConfig) -> Mapping[str, object]:
    resolved = OmegaConf.to_container(value, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("Resolved paths configuration must be a mapping.")
    return cast(Mapping[str, object], resolved)


def _renderer_command(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise TypeError("renderer.command must be a list of non-empty strings.")
    return tuple(value)


@hydra_main(
    version_base="1.3",
    config_path="../../configs",
    config_name="dataset/pipeline",
    validation_boundary="synthetic.dataset.pipeline",
)
def main(cfg: DictConfig) -> int:
    """Write the shared path manifest and optionally execute every stage."""
    runtime = validate_config("synthetic.dataset.pipeline", cfg)
    manifest = PathPipelineManifest.from_config(
        _mapping(cfg.paths),
        resolver=runtime.resolver,
    )
    execute = cast(bool, runtime.values["execute"])
    if not execute:
        manifest.write()
        return 0
    command = _renderer_command(
        OmegaConf.to_container(cfg.renderer.command, resolve=True)
    )
    working_directory = runtime.path(
        PathRole.EXTERNAL_ASSET,
        "renderer.working_directory",
    )
    renderer = runtime.values["renderer"]
    if not isinstance(renderer, Mapping):
        raise AssertionError("Validated renderer configuration is not a mapping.")
    renderer_mode = cast(str, renderer["mode"])
    validate_pipeline_inputs(
        manifest,
        renderer_mode=renderer_mode,
        renderer_command=command,
        working_directory=working_directory,
    )
    manifest.write()
    execute_pipeline(
        manifest,
        renderer_mode=renderer_mode,
        renderer_command=command,
        working_directory=working_directory,
    )
    return 0


if __name__ == "__main__":
    cast(Any, main)()
