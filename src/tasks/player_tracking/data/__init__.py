"""Synthetic multi-person data contracts."""

from src.tasks.player_tracking.data.datamodule import PlayerTrackingDataModule
from src.tasks.player_tracking.data.synthetic import SyntheticPlayerTrackingDataset
from src.tasks.player_tracking.data.types import (
    PlayerTrackingBatch,
    PlayerTrackingPrediction,
)

__all__ = [
    "PlayerTrackingBatch",
    "PlayerTrackingDataModule",
    "PlayerTrackingPrediction",
    "SyntheticPlayerTrackingDataset",
]
