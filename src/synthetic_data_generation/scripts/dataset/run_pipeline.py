"""
Plan or execute one registered 3DGS-native synthetic-dataset pipeline.

Usage:
    python -m src.synthetic_data_generation.scripts.dataset.run_pipeline domain=blcs
    python -m src.synthetic_data_generation.scripts.dataset.run_pipeline domain=court execute=true

Notes:
    - Hydra loads configs from `src/synthetic_data_generation/configs/dataset`.
    - `execute=false` publishes only the immutable command plan.
    - NHT stages run in the independently pinned `third_party/nht/.venv`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src.synthetic_data_generation.dataset.execution import DatasetPipelineExecutor
from src.synthetic_data_generation.dataset.registry import get_dataset_pipeline
from src.synthetic_data_generation.rendering.nht.process import (
    NhtProcessBackend,
    NhtRuntime,
)
from src.utils.hydra import hydra_main


def _mapping(config: DictConfig) -> Mapping[str, object]:
    value = OmegaConf.to_container(config, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("Resolved dataset configuration must be a mapping.")
    return cast(Mapping[str, object], value)


def _path(value: object) -> Path:
    return Path(to_absolute_path(str(value)))


@hydra_main(
    version_base="1.3",
    config_path="../../configs/dataset",
    config_name="pipeline",
)
def main(cfg: DictConfig) -> int:
    """Build the selected domain plan and optionally execute every stage."""
    domain_name = str(cfg.domain.name)
    pipeline = get_dataset_pipeline(domain_name)
    plan = pipeline.build_plan(_mapping(cfg.domain))
    plan_path = _path(cfg.plan_path)
    if plan_path.exists():
        raise FileExistsError(f"Dataset pipeline plan refuses overwrite: {plan_path}")
    plan.write(plan_path)
    if not bool(cfg.execute):
        return 0

    runtime = NhtRuntime(
        repository=_path(cfg.runtime.repository),
        python=_path(cfg.runtime.python),
        expected_commit=str(cfg.runtime.expected_commit),
        require_clean=bool(cfg.runtime.require_clean),
    )
    backend = NhtProcessBackend(
        project_root=_path("."),
        runtime=runtime,
    )
    executor = DatasetPipelineExecutor(
        project_root=_path("."),
        nht_backend=backend,
    )
    executor.execute(plan, output_dir=_path(cfg.execution_output))
    return 0


if __name__ == "__main__":
    main()
