"""DataModule for court keypoint detection."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.tasks.court_detection.data.dataset import CourtKeypointDataset

if TYPE_CHECKING:
    from omegaconf import DictConfig


class CourtKeypointDataModule(pl.LightningDataModule):
    """Lightning DataModule for court keypoint detection.

    Args:
        config: Configuration dictionary with data parameters.
    """

    def __init__(self, config: DictConfig | None = None) -> None:
        """Initialize the DataModule.

        Args:
            config: Configuration dictionary with data parameters.

        """
        super().__init__()
        self.config = config or {}

        data_cfg = self.config.get("data", {})
        self.data_dir = Path(data_cfg.get("data_dir", "data/court_detection/scenes"))
        self.batch_size = data_cfg.get("batch_size", 32)
        self.num_workers = data_cfg.get("num_workers", 4)
        self.pin_memory = data_cfg.get("pin_memory", True)
        self.input_size = tuple(data_cfg.get("input_size", [256, 256]))
        self.heatmap_size = tuple(data_cfg.get("heatmap_size", [64, 64]))
        self.train_split = data_cfg.get("train_split", 0.8)
        self.val_split = data_cfg.get("val_split", 0.1)
        self.test_split = data_cfg.get("test_split", 0.1)
        self.augmentation = data_cfg.get("augmentation", {})

        self.train_dataset: CourtKeypointDataset | None = None
        self.val_dataset: CourtKeypointDataset | None = None
        self.test_dataset: CourtKeypointDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for each stage."""
        if stage == "fit" or stage is None:
            self.train_dataset = CourtKeypointDataset(
                data_dir=self.data_dir,
                split="train",
                input_size=self.input_size,
                heatmap_size=self.heatmap_size,
                augmentation=self.augmentation,
            )
            self.val_dataset = CourtKeypointDataset(
                data_dir=self.data_dir,
                split="val",
                input_size=self.input_size,
                heatmap_size=self.heatmap_size,
            )

        if stage == "test" or stage is None:
            self.test_dataset = CourtKeypointDataset(
                data_dir=self.data_dir,
                split="test",
                input_size=self.input_size,
                heatmap_size=self.heatmap_size,
            )

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
