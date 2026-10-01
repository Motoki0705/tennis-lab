"""Data interfaces for ball detection."""

from typing import Any

import pytorch_lightning as pl

from src.tasks.ball_detection.configuration import validate_data
from src.tasks.ball_detection.data.components.augmentation import (
    BallDetectionAugmentation,
)
from src.tasks.ball_detection.data.dataset import (
    BallDetectionDataset,
    WindowFrame,
    WindowFrames,
)
from src.tasks.ball_detection.data.staged_datamodule import StagedBallDataModule
from src.tasks.ball_detection.data.store_datamodule import BallStoreDataModule
from src.tasks.ball_detection.data.store_dataset import BallStoreDataset
from src.tasks.ball_detection.data.types import (
    BallDetectionBatch,
    BallDetectionSample,
    FrameLabel,
)
from src.tasks.ball_detection.data.web_datamodule import (
    WebBallDataModule,
    WebBallDetectionDataset,
)


def build_ball_detection_datamodule(config: Any) -> pl.LightningDataModule:
    """Build the configured dataset-specific DataModule."""
    source = str(validate_data(config)["source"])
    datamodule_types: dict[str, type[pl.LightningDataModule]] = {
        "store": BallStoreDataModule,
        "web": WebBallDataModule,
        "staged": StagedBallDataModule,
    }
    try:
        datamodule_type = datamodule_types[source]
    except KeyError as error:
        supported = ", ".join(sorted(datamodule_types))
        raise ValueError(
            f"Unsupported ball detection data.source={source!r}; expected {supported}."
        ) from error
    return datamodule_type(config)


__all__ = [
    "BallDetectionAugmentation",
    "BallDetectionBatch",
    "BallDetectionDataset",
    "BallDetectionSample",
    "BallStoreDataModule",
    "BallStoreDataset",
    "FrameLabel",
    "StagedBallDataModule",
    "WebBallDataModule",
    "WebBallDetectionDataset",
    "WindowFrame",
    "WindowFrames",
    "build_ball_detection_datamodule",
]
