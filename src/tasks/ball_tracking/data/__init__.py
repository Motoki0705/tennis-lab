"""Data contracts and synthetic data for multi-ball tracking."""

from src.tasks.ball_tracking.data.datamodule import BallTrackingDataModule
from src.tasks.ball_tracking.data.synthetic import SyntheticBallTrackingDataset
from src.tasks.ball_tracking.data.types import BallTrackingBatch, BallTrackingPrediction

__all__ = [
    "BallTrackingBatch",
    "BallTrackingDataModule",
    "BallTrackingPrediction",
    "SyntheticBallTrackingDataset",
]
