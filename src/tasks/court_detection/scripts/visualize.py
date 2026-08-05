"""Visualize court detection predictions (kp / seg / line) for one image source.

Usage:
    python -m src.tasks.court_detection.scripts.visualize
    python -m src.tasks.court_detection.scripts.visualize visualization=seg
    python -m src.tasks.court_detection.scripts.visualize visualization=line \
        visualization.image_source=court/images/foo.png

Notes:
    - Hydra loads configuration from `src/tasks/court_detection/configs/visualize.yaml`.
    - Select the task with `visualization=kp|seg|line`; each saves a 2-panel GIF.
"""

from __future__ import annotations

import sys

from omegaconf import DictConfig

from src.tasks.court_detection.visualization.orchestrator import (
    build_runtime_config,
    run_visualization,
)
from src.utils.hydra import hydra_main, register_boundary_validator

_BOUNDARY = "court_detection.visualize"


def _validate_boundary(config: DictConfig) -> None:
    build_runtime_config(config)


register_boundary_validator(_BOUNDARY, _validate_boundary)


@hydra_main(
    config_path="../configs",
    config_name="visualize",
    version_base="1.3",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Hydra entry point."""
    runtime = build_runtime_config(cfg)
    return run_visualization(runtime)


if __name__ == "__main__":
    sys.exit(main())
