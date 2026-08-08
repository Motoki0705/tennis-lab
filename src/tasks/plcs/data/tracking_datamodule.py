"""Canonical compact PLCS tracking data module."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.data.tracking_dataset import (
    PLCSTrackingDataset,
    collate_plcs_tracking_batch,
)


class PLCSTrackingDataModule(pl.LightningDataModule):
    """Build lifecycle samples directly from canonical manifest splits."""

    def __init__(self, config: object) -> None:
        super().__init__()
        self.runtime = PLCSTrainingConfig.from_config(config)
        self.train_dataset: PLCSTrackingDataset | None = None
        self.val_dataset: PLCSTrackingDataset | None = None
        self.test_dataset: PLCSTrackingDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        directory = self.runtime.data.dataset_dir
        config = self.runtime.raw
        if stage in {None, "fit"}:
            self.train_dataset = PLCSTrackingDataset(
                dataset_dir=directory, split="train", config=config, augment=True
            )
            self.val_dataset = PLCSTrackingDataset(
                dataset_dir=directory,
                split="validation",
                config=config,
                augment=False,
            )
        if stage in {None, "test"}:
            self.test_dataset = PLCSTrackingDataset(
                dataset_dir=directory, split="test", config=config, augment=False
            )

    def _loader(
        self, dataset: PLCSTrackingDataset, *, shuffle: bool
    ) -> DataLoader[Any]:
        data = self.runtime.data
        return DataLoader(
            dataset,
            batch_size=data.batch_size,
            shuffle=shuffle,
            num_workers=data.num_workers,
            pin_memory=data.pin_memory,
            collate_fn=collate_plcs_tracking_batch,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        if self.train_dataset is None:
            raise RuntimeError("PLCSTrackingDataModule.setup('fit') must run first.")
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader[Any]:
        if self.val_dataset is None:
            raise RuntimeError("PLCSTrackingDataModule.setup('fit') must run first.")
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader[Any]:
        if self.test_dataset is None:
            raise RuntimeError("PLCSTrackingDataModule.setup('test') must run first.")
        return self._loader(self.test_dataset, shuffle=False)


__all__ = ["PLCSTrackingDataModule"]
