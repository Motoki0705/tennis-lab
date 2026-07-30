"""
Plan or execute the generic path-driven synthetic-data pipeline.

Usage:
    python -m src.synthetic_data_generation.scripts.dataset.run_pipeline
    python -m src.synthetic_data_generation.scripts.dataset.run_pipeline execute=true

Notes:
    - Hydra loads configuration from `src/synthetic_data_generation/configs/dataset`.
    - Empty `renderer.command` uses passthrough copies for prepared render inputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from src.synthetic_data_generation.dataset.execution import execute_pipeline
from src.synthetic_data_generation.dataset.pipeline import PathPipelineManifest
from src.utils.hydra import hydra_main
from src.utils.paths import PROJECT_ROOT


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
    config_path="../../configs/dataset",
    config_name="pipeline",
)
def main(cfg: DictConfig) -> int:
    """Write the shared path manifest and optionally execute every stage."""
    project_root = Path(str(cfg.project_root))
    if not project_root.is_absolute():
        project_root = PROJECT_ROOT / project_root
    project_root = project_root.resolve()
    manifest = PathPipelineManifest.from_config(
        _mapping(cfg.paths),
        project_root=project_root,
    )
    manifest.write()
    if not bool(cfg.execute):
        return 0
    command = _renderer_command(
        OmegaConf.to_container(cfg.renderer.command, resolve=True)
    )
    working_directory = Path(str(cfg.renderer.working_directory))
    if not working_directory.is_absolute():
        working_directory = project_root / working_directory
    execute_pipeline(
        manifest,
        renderer_command=command,
        working_directory=working_directory,
    )
    return 0


if __name__ == "__main__":
    cast(Any, main)()
