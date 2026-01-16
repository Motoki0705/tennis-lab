"""PyTorch Lightning DataModule for MAE training.

This DataModule reads preprocessed batches from a cache directory. Caches are
produced asynchronously by `MAEEpochCacheCallback`, ensuring:
  - batch-internal fixed resolution (no padding),
  - minimal `Dataset` overhead (load + return),
  - low-resolution-heavy sampling via bucket planning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.mae.data.cache.paths import EpochCachePaths
from src.mae.data.dataset_cached import CachedBatchIterableDataset, CachedBatchSource


class MAEDataModule(pl.LightningDataModule):
    """Lightning DataModule for cached-batch MAE pre-training."""

    def __init__(
        self,
        cache_root: str | Path = "data/mae/cache",
        num_workers: int = 4,
        val_split: float = 0.1,
        pin_memory: bool = True,
        cache_map_location: str = "cpu",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.cache_root = Path(cache_root)
        self.num_workers = int(num_workers)
        self.val_split = float(val_split)
        self.pin_memory = bool(pin_memory)
        self.cache_map_location = str(cache_map_location)

        self.train_dataset: CachedBatchIterableDataset | None = None
        self.val_dataset: CachedBatchIterableDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            train_paths = EpochCachePaths(cache_root=self.cache_root, split="train")
            self.train_dataset = CachedBatchIterableDataset(
                CachedBatchSource(manifest_pointer_path=train_paths.current_pointer_path()),
                map_location=self.cache_map_location,
            )

            if self.val_split > 0:
                val_paths = EpochCachePaths(cache_root=self.cache_root, split="val")
                self.val_dataset = CachedBatchIterableDataset(
                    CachedBatchSource(manifest_path=val_paths.val_manifest_path()),
                    map_location=self.cache_map_location,
                )
            else:
                self.val_dataset = None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=None,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            return DataLoader([])
        return DataLoader(
            self.val_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=None,
        )

    @classmethod
    def from_config(cls, config: Any) -> "MAEDataModule":
        data_cfg = config.get("data", config)
        return cls(
            cache_root=data_cfg.get("cache_root", "data/mae/cache"),
            num_workers=data_cfg.get("num_workers", 4),
            val_split=data_cfg.get("val_split", 0.1),
            pin_memory=data_cfg.get("pin_memory", True),
            cache_map_location=data_cfg.get("cache_map_location", "cpu"),
        )


if __name__ == "__main__":  # pragma: no cover
    dm = MAEDataModule()
    print(dm)
