"""Rank retained proposal/annotation pairs or render one explicit interval."""

from __future__ import annotations

import sys

from omegaconf import DictConfig

from src.tasks.ball_detection.visualization.presentation.annotation_correction import (
    run_visualization,
)
from src.tasks.ball_detection.visualization.presentation.annotation_correction_config import (
    build_runtime_config,
)
from src.utils.hydra import hydra_main, register_boundary_validator


def _validate_boundary(config: DictConfig) -> None:
    build_runtime_config(config)


register_boundary_validator("ball.annotation_correction", _validate_boundary)


@hydra_main(
    config_path="../configs",
    config_name="visualize_annotation_correction",
    version_base="1.3",
    validation_boundary="ball.annotation_correction",
)
def main(config: DictConfig) -> int:
    return run_visualization(build_runtime_config(config))


if __name__ == "__main__":
    sys.exit(main())
