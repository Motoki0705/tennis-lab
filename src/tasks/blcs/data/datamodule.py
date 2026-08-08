"""Canonical compact BLCS Lightning data module."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.tasks.blcs.data.dataset import BallTrajectoryDataset
from src.utils.paths import PROJECT_ROOT


class BLCSDataModule(pl.LightningDataModule):
    """Build split datasets directly from one canonical ``dataset.json``."""

    def __init__(self, config: Any, *, collate_fn: Callable[..., Any]) -> None:
        super().__init__()
        self.config = config
        self.collate_fn = collate_fn
        self.train_dataset: BallTrajectoryDataset | None = None
        self.val_dataset: BallTrajectoryDataset | None = None
        self.test_dataset: BallTrajectoryDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        directory = PROJECT_ROOT / "data" / str(self.config.data.dataset_dir)
        if stage in {None, "fit"}:
            self.train_dataset = BallTrajectoryDataset(
                dataset_dir=directory, split="train", config=self.config, augment=True
            )
            self.val_dataset = BallTrajectoryDataset(
                dataset_dir=directory,
                split="validation",
                config=self.config,
                augment=False,
            )
        if stage in {None, "test"}:
            self.test_dataset = BallTrajectoryDataset(
                dataset_dir=directory, split="test", config=self.config, augment=False
            )

    def _loader(
        self, dataset: BallTrajectoryDataset, *, shuffle: bool
    ) -> DataLoader[Any]:
        return DataLoader(
            dataset,
            batch_size=int(self.config.data.batch_size),
            shuffle=shuffle,
            num_workers=int(self.config.data.num_workers),
            pin_memory=bool(self.config.data.pin_memory),
            collate_fn=self.collate_fn,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        if self.train_dataset is None:
            raise RuntimeError("BLCSDataModule.setup('fit') must run first.")
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader[Any]:
        if self.val_dataset is None:
            raise RuntimeError("BLCSDataModule.setup('fit') must run first.")
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader[Any]:
        if self.test_dataset is None:
            raise RuntimeError("BLCSDataModule.setup('test') must run first.")
        return self._loader(self.test_dataset, shuffle=False)


__all__ = ["BLCSDataModule"]
