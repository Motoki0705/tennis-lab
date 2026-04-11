"""Data interfaces for ball detection."""

from src.tasks.ball_detection.data.argumentation import BallDetectionArgumentation
from src.tasks.ball_detection.data.datamodule import BallDetectionDataModule
from src.tasks.ball_detection.data.dataset import BallDetectionDataset
from src.tasks.ball_detection.data.types import BallDetectionBatch, BallDetectionSample

__all__ = [
    "BallDetectionArgumentation",
    "BallDetectionBatch",
    "BallDetectionDataModule",
    "BallDetectionDataset",
    "BallDetectionSample",
]
