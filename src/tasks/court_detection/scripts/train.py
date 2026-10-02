"""Train all Court outputs from Synthetic V3 and TennisCourtDetector images."""

from __future__ import annotations

from omegaconf import DictConfig

from src.tasks.court_detection.training.runner import (
    CourtDetectionTrainingRunner,
    validate_train_boundary,
)
from src.utils.hydra import hydra_main, register_boundary_validator

_BOUNDARY = "court_detection.train"
register_boundary_validator(_BOUNDARY, validate_train_boundary)


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="train",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> None:
    """Train court detection model."""
    runner = CourtDetectionTrainingRunner()
    runner.run(cfg)


if __name__ == "__main__":
    main()
