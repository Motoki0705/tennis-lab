"""Visualize PLCS outputs with Hydra-managed configuration.

Usage:
    python -m src.tasks.plcs.scripts.visualize
    python -m src.tasks.plcs.scripts.visualize run.output_dir=plcs/visualization

Notes:
    - Configuration is loaded from `src/tasks/plcs/configs/visualize.yaml`.
    - The script uses Hydra for configuration loading.
"""

from __future__ import annotations

import sys

from omegaconf import DictConfig

from src.tasks.plcs.visualization.orchestrator import (
    build_runtime_config,
    run_visualization,
)
from src.utils.hydra import hydra_main


@hydra_main(
    config_path="../configs",
    config_name="visualize",
    version_base="1.3",
    validation_boundary="plcs.visualize",
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Hydra entry point."""
    runtime = build_runtime_config(cfg)
    return int(run_visualization(runtime))


if __name__ == "__main__":
    sys.exit(main())
