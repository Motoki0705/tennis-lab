"""Lightning data module for on-the-fly multi-person synthetic scenes."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.tasks.player_tracking.data.synthetic import SyntheticPlayerTrackingDataset


class PlayerTrackingDataModule(pl.LightningDataModule):
    """Build deterministic train/validation/test multi-person clips."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.data_config = config.data
        self.batch_size = int(self.data_config.batch_size)
        self.num_workers = int(self.data_config.num_workers)
        self.train_dataset: SyntheticPlayerTrackingDataset | None = None
        self.val_dataset: SyntheticPlayerTrackingDataset | None = None
        self.test_dataset: SyntheticPlayerTrackingDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in {None, "fit"}:
            self.train_dataset = SyntheticPlayerTrackingDataset(
                self.data_config, split="train"
            )
            self.val_dataset = SyntheticPlayerTrackingDataset(
                self.data_config, split="val"
            )
        if stage in {None, "test"}:
            self.test_dataset = SyntheticPlayerTrackingDataset(
                self.data_config, split="test"
            )

    def _loader(
        self, dataset: SyntheticPlayerTrackingDataset | None, *, shuffle: bool
    ) -> DataLoader[dict[str, Any]]:
        if dataset is None:
            raise RuntimeError("setup() must be called before requesting a dataloader.")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader[dict[str, Any]]:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader[dict[str, Any]]:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader[dict[str, Any]]:
        return self._loader(self.test_dataset, shuffle=False)
