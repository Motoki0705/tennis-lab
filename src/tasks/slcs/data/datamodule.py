"""Lightning DataModule for SLCS training on the issue #634 dataset."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.tasks.slcs.configuration import SLCSDataRuntimeConfig
from src.tasks.slcs.data.dataset import SLCSWindowDataset, collate_slcs


class SLCSDataModule(pl.LightningDataModule):
    """Builds train/val/test :class:`SLCSWindowDataset` loaders.

    Required ``config.data`` keys: ``dataset_root``, ``split_file``,
    ``batch_size`` and the :class:`SLCSDataConfig` fields (``window_size``,
    ``dino`` section, ...).
    """

    def __init__(self, config: SLCSDataRuntimeConfig) -> None:
        super().__init__()
        self.config = config
        self.dataset_root = config.dataset_root
        self.split_file = config.split_file
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.pin_memory = config.pin_memory
        self.overfit = config.overfit
        self.data_config = config.pipeline

        self.train_dataset: SLCSWindowDataset | None = None
        self.val_dataset: SLCSWindowDataset | None = None
        self.test_dataset: SLCSWindowDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = self._build_dataset("train")
            self.val_dataset = self._build_dataset("val")
        if stage == "validate" and self.val_dataset is None:
            self.val_dataset = self._build_dataset("val")
        if stage in (None, "test") and self.test_dataset is None:
            self.test_dataset = self._build_dataset("test")

    def _build_dataset(self, split: str) -> SLCSWindowDataset:
        source_split = "train" if self.overfit else split
        return SLCSWindowDataset(
            dataset_root=self.dataset_root,
            split_file=self.split_file,
            split=source_split,
            config=self.data_config,
            stride=(
                self.data_config.train_stride
                if source_split == "train"
                else self.data_config.eval_stride
            ),
        )

    def _loader(self, dataset: SLCSWindowDataset, *, shuffle: bool) -> DataLoader[Any]:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_slcs,
            drop_last=False,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        if self.train_dataset is None:
            raise RuntimeError("setup('fit') must run before train_dataloader().")
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader[Any]:
        if self.val_dataset is None:
            raise RuntimeError("setup('fit') must run before val_dataloader().")
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader[Any]:
        if self.test_dataset is None:
            raise RuntimeError("setup('test') must run before test_dataloader().")
        return self._loader(self.test_dataset, shuffle=False)


__all__ = ["SLCSDataModule"]
