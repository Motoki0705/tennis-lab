"""Train a ball detection model with PyTorch Lightning.

Usage:
    python -m src.tasks.ball_detection.training.convnext_heatmap.cli
    python -m src.tasks.ball_detection.training.convnext_heatmap.cli run.dry_run=true
    python -m src.tasks.ball_detection.training.convnext_heatmap.cli training.trainer.max_epochs=1

Notes:
    - Hydra loads configuration from `src/tasks/ball_detection/configs/train.yaml`.
    - The script forwards the resolved config to the training runner.
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.tasks.ball_detection import configuration as _configuration  # noqa: F401
from src.tasks.ball_detection.training.convnext_heatmap.runner import (
    BallDetectionTrainingRunner,
)
from src.utils.hydra import hydra_main


@hydra_main(
    version_base="1.3",
    config_path="../../configs",
    config_name="train",
    validation_boundary="ball.train",
)
def main(cfg: DictConfig) -> None:
    """Train ball detection model."""
    runner = BallDetectionTrainingRunner()
    runner.run(cfg)
