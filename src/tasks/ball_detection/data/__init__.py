"""Data interfaces for ball detection."""

from typing import Any

from src.tasks.ball_detection.configuration import validate_data
from src.tasks.ball_detection.data.components.augmentation import (
    BallDetectionAugmentation,
)
from src.tasks.ball_detection.data.dataset import (
    BallDetectionDataset,
    WindowFrame,
    WindowFrames,
)
from src.tasks.ball_detection.data.store_datamodule import BallStoreDataModule
from src.tasks.ball_detection.data.store_dataset import BallStoreDataset
from src.tasks.ball_detection.data.types import (
    BallDetectionBatch,
    BallDetectionSample,
    FrameLabel,
)


def build_ball_detection_datamodule(config: Any) -> BallStoreDataModule:
    """Build the unified ball frame store DataModule."""
    validate_data(config)
    return BallStoreDataModule(config)


__all__ = [
    "BallDetectionAugmentation",
    "BallDetectionBatch",
    "BallDetectionDataset",
    "BallDetectionSample",
    "BallStoreDataModule",
    "BallStoreDataset",
    "FrameLabel",
    "WindowFrame",
    "WindowFrames",
    "build_ball_detection_datamodule",
]
