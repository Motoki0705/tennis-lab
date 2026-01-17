"""DataModule for court keypoint detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.court_detection.data.dataset import CourtKeypointDataset


class CourtKeypointDataModule(pl.LightningDataModule):
    """Lightning DataModule for court keypoint detection.

    Args:
        data_dir: Path to data directory.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory.
        input_size: Input image size [H, W].
        heatmap_size: Output heatmap size [H, W].
        train_split: Training split ratio.
        val_split: Validation split ratio.
        test_split: Test split ratio.
        augmentation: Augmentation config dict.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/court_detection/scenes",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        input_size: list[int] | tuple[int, int] = (256, 256),
        heatmap_size: list[int] | tuple[int, int] = (64, 64),
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        augmentation: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.input_size = tuple(input_size)
        self.heatmap_size = tuple(heatmap_size)
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.augmentation = augmentation or {}

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
