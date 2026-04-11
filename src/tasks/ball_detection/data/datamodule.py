"""Lightning DataModule for ball detection."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.tasks.ball_detection.data.argumentation import BallDetectionArgumentation
from src.tasks.ball_detection.data.dataset import BallDetectionDataset

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BallDetectionDataModule(pl.LightningDataModule):
    """Lightning DataModule for ball detection.

    Wraps :class:`BallDetectionDataset` with train/val/test splits,
    augmentation setup, and DataLoader construction.

    Args:
        config: Full Hydra configuration dictionary.
    """

    def __init__(self, config: DictConfig | None = None) -> None:
        super().__init__()
        self.config = config or {}

        data_cfg = self.config.get("data", {})
        self.data_dir = Path(str(data_cfg.get("data_dir", "data/tennis")))
        self.batch_size = int(data_cfg.get("batch_size", 4))
        self.num_workers = int(data_cfg.get("num_workers", 4))
        self.pin_memory = bool(data_cfg.get("pin_memory", True))

        split_cfg = data_cfg.get("split", {})
        self.train_split_file = str(split_cfg.get("train_file", ""))
        self.val_split_file = str(split_cfg.get("val_file", ""))
        self.test_split_file = str(split_cfg.get("test_file", ""))

        self.train_dataset: BallDetectionDataset | None = None
        self.val_dataset: BallDetectionDataset | None = None
        self.test_dataset: BallDetectionDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for each stage."""
        aug_cfg = self.config.get("data", {}).get("augmentation", {})

        if stage == "fit" or stage is None:
            train_aug = BallDetectionArgumentation(aug_cfg)
            self.train_dataset = BallDetectionDataset(
                data_dir=self.data_dir,
                split_file=self.train_split_file,
                config=self.config,
                argumentation=train_aug,
            )
            self.val_dataset = BallDetectionDataset(
                data_dir=self.data_dir,
                split_file=self.val_split_file,
                config=self.config,
                argumentation=BallDetectionArgumentation.from_eval_config(aug_cfg),
            )

        if stage == "test" or stage is None:
            self.test_dataset = BallDetectionDataset(
                data_dir=self.data_dir,
                split_file=self.test_split_file,
                config=self.config,
                argumentation=BallDetectionArgumentation.from_eval_config(aug_cfg),
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
