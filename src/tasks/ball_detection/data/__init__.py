"""Data interfaces for ball detection."""

from typing import Any

from src.tasks.ball_detection.data.argumentation import BallDetectionArgumentation
from src.tasks.ball_detection.data.dataset import BallDetectionDataset
from src.tasks.ball_detection.data.tracknet_datamodule import TrackNetDataModule
from src.tasks.ball_detection.data.types import (
    BallDetectionBatch,
    BallDetectionSample,
    ClipWindow,
    FrameLabel,
)
from src.tasks.ball_detection.data.youtube_datamodule import YouTubeDataModule


def build_ball_detection_datamodule(config: Any) -> TrackNetDataModule:
    """Build the configured dataset-specific DataModule."""
    source = str(config.get("data", {}).get("source", "tracknet")).lower()
    datamodule_types = {
        "tracknet": TrackNetDataModule,
        "youtube": YouTubeDataModule,
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
    "BallDetectionArgumentation",
    "BallDetectionBatch",
    "BallDetectionDataset",
    "BallDetectionSample",
    "ClipWindow",
    "FrameLabel",
    "TrackNetDataModule",
    "YouTubeDataModule",
    "build_ball_detection_datamodule",
]
