"""Data interfaces for ball detection."""

from typing import Any

import pytorch_lightning as pl

from src.tasks.ball_detection.data.components.augmentation import (
    BallDetectionAugmentation,
)
from src.tasks.ball_detection.data.dataset import BallDetectionDataset
from src.tasks.ball_detection.data.mixed_tracknet_datamodule import (
    MixedTrackNetDataModule,
)
from src.tasks.ball_detection.data.staged_datamodule import StagedBallDataModule
from src.tasks.ball_detection.data.tracknet_datamodule import TrackNetDataModule
from src.tasks.ball_detection.data.types import (
    BallDetectionBatch,
    BallDetectionSample,
    ClipWindow,
    FrameLabel,
)
from src.tasks.ball_detection.data.web_datamodule import (
    WebBallDataModule,
    WebBallDetectionDataset,
)
from src.tasks.ball_detection.data.youtube_datamodule import YouTubeDataModule


def build_ball_detection_datamodule(config: Any) -> pl.LightningDataModule:
    """Build the configured dataset-specific DataModule."""
    source = str(config.get("data", {}).get("source", "tracknet")).lower()
    datamodule_types: dict[str, type[pl.LightningDataModule]] = {
        "tracknet": TrackNetDataModule,
        "mixed_tracknet": MixedTrackNetDataModule,
        "youtube": YouTubeDataModule,
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
    "ClipWindow",
    "FrameLabel",
    "MixedTrackNetDataModule",
    "StagedBallDataModule",
    "TrackNetDataModule",
    "WebBallDataModule",
    "WebBallDetectionDataset",
    "YouTubeDataModule",
    "build_ball_detection_datamodule",
]
