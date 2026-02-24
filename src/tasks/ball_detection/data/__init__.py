"""Datasets and datamodules for ball_detection."""

from src.tasks.ball_detection.data.datamodule import BallDetectionDataModule
from src.tasks.ball_detection.data.pseudo_datamodule import BallDetectionPseudoDataModule

__all__ = ["BallDetectionDataModule", "BallDetectionPseudoDataModule"]
