"""Composable data loading for Court detection."""

from src.tasks.court_detection.data.datamodule import CourtDetectionDataModule
from src.tasks.court_detection.data.dataset import CourtDetectionDataset

__all__ = ["CourtDetectionDataModule", "CourtDetectionDataset"]
