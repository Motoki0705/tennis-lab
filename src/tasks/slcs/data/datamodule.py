"""Lightning DataModule for SLCS training on the issue #634 dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.tasks.slcs.data.dataset import SLCSDataConfig, SLCSWindowDataset, collate_slcs

if TYPE_CHECKING:
    from omegaconf import DictConfig


class SLCSDataModule(pl.LightningDataModule):
    """Builds train/val/test :class:`SLCSWindowDataset` loaders.

    Required ``config.data`` keys: ``dataset_root``, ``split_file``,
    ``batch_size`` and the :class:`SLCSDataConfig` fields (``window_size``,
    ``dino`` section, ...).
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        data_cfg = config.get("data")
        if data_cfg is None:
            raise ValueError("config must contain a 'data' section.")
        dataset_root = data_cfg.get("dataset_root")
        split_file = data_cfg.get("split_file")
        if not dataset_root or not split_file:
            raise ValueError("config.data must set dataset_root and split_file.")
        self.dataset_root = str(dataset_root)
        self.split_file = str(split_file)
        self.batch_size = int(data_cfg.get("batch_size", 8))
        self.num_workers = int(data_cfg.get("num_workers", 0))
        self.overfit = bool(data_cfg.get("overfit", False))
        self.data_config = SLCSDataConfig.from_config(data_cfg)

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
        )

    def _loader(self, dataset: SLCSWindowDataset, *, shuffle: bool) -> DataLoader[Any]:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_slcs,
            drop_last=False,
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
