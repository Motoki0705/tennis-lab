"""Training entry points for ball_detection."""

from src.tasks.ball_detection.training.lightning_module import BallDetectionLightningModule
from src.tasks.ball_detection.training.runner import BallDetectionTrainingRunner

__all__ = ["BallDetectionLightningModule", "BallDetectionTrainingRunner"]
