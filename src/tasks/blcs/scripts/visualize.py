"""Visualize BLCS reconstruction or inference outputs.

Usage:
    python -m src.tasks.blcs.scripts.visualize
    python -m src.tasks.blcs.scripts.visualize visualization.save=blcs/preview.mp4

Notes:
    - Hydra loads configuration from `src/tasks/blcs/configs/visualize.yaml`.
    - The script builds a runtime visualization config and forwards it to the orchestrator.
"""

from __future__ import annotations

import sys

from omegaconf import DictConfig

from src.tasks.blcs.configuration import validate_visualization_boundary
from src.tasks.blcs.visualization.orchestrator import (
    build_runtime_config,
    run_visualization,
)
from src.utils.hydra import hydra_main


@hydra_main(
    config_path="../configs",
    config_name="visualize",
    version_base="1.3",
    validation_boundary="blcs.visualize",
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Hydra entry point."""
    validate_visualization_boundary(cfg)
    runtime = build_runtime_config(cfg)
    return run_visualization(runtime)


if __name__ == "__main__":
    sys.exit(main())
