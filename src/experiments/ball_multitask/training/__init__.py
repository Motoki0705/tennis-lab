"""Training components for ball multi-task learning."""

from src.experiments.ball_multitask.training.lightning_module import BallMultitaskLightningModule
from src.experiments.ball_multitask.training.runner import BallMultitaskTrainingRunner

__all__ = ["BallMultitaskLightningModule", "BallMultitaskTrainingRunner"]
