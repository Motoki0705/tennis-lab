"""Train a court detection model with PyTorch Lightning.

Usage:
    python -m src.tasks.court_detection.scripts.train
    python -m src.tasks.court_detection.scripts.train data=court_kp loss=kp
    python -m src.tasks.court_detection.scripts.train data=court_line loss=line
    python -m src.tasks.court_detection.scripts.train run.dry_run=true

Notes:
    - Hydra loads configuration from ``src/tasks/court_detection/configs/train.yaml``.
    - ``model`` defaults to the config matching the selected ``data`` group.
    - When you switch tasks, set ``loss`` to the same task family.
    - The script forwards the resolved config to the training runner.
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.tasks.court_detection.configuration import validate_train_boundary
from src.tasks.court_detection.training.runner import CourtDetectionTrainingRunner
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
