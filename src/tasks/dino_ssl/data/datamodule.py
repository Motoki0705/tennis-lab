"""Lightning datamodule for DINOv3 tennis SSL."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader, Dataset, random_split

from src.tasks.dino_ssl.data.augmentation import (
    DataAugmentationDINO,
    MaskingGenerator,
)
from src.tasks.dino_ssl.data.dataset import SSLImageDataset, ssl_collate_fn


class DinoSSLDataModule(pl.LightningDataModule):
    """Build train/val multi-crop SSL dataloaders from a collected manifest."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        data_cfg = config.data
        self.root = to_absolute_path(str(data_cfg.root))
        self.batch_size = int(data_cfg.batch_size)
        self.num_workers = int(data_cfg.num_workers)
        self.pin_memory = bool(data_cfg.get("pin_memory", True))
        self.val_fraction = float(data_cfg.get("val_fraction", 0.1))
        self.seed = int(config.run.get("seed", 42))

        aug_cfg = data_cfg.augmentation
        self.augmentation = DataAugmentationDINO(
            global_crops_scale=tuple(aug_cfg.global_crops_scale),
            local_crops_scale=tuple(aug_cfg.local_crops_scale),
            local_crops_number=int(aug_cfg.local_crops_number),
            global_size=int(aug_cfg.global_size),
            local_size=int(aug_cfg.local_size),
        )
        self.masking = MaskingGenerator(
            input_size=int(aug_cfg.global_size),
            patch_size=int(data_cfg.patch_size),
            mask_ratio_min=float(aug_cfg.mask_ratio_min),
            mask_ratio_max=float(aug_cfg.mask_ratio_max),
        )
        self._train_dataset: Dataset | None = None
        self._val_dataset: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        dataset = SSLImageDataset(
            root=self.root,
            augmentation=self.augmentation,
            masking=self.masking,
        )
        val_len = int(round(len(dataset) * self.val_fraction))
        val_len = min(max(val_len, 1), len(dataset) - 1) if len(dataset) > 1 else 0
        train_len = len(dataset) - val_len
        if val_len > 0:
            import torch

            generator = torch.Generator().manual_seed(self.seed)
            self._train_dataset, self._val_dataset = random_split(
                dataset, [train_len, val_len], generator=generator
            )
        else:
            self._train_dataset = dataset
            self._val_dataset = None

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=shuffle,
            collate_fn=ssl_collate_fn,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        assert self._train_dataset is not None
        return self._loader(self._train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader | None:
        if self._val_dataset is None:
            return None
        return self._loader(self._val_dataset, shuffle=False)


def build_dino_ssl_datamodule(config: Any) -> DinoSSLDataModule:
    """Factory used by the training runner."""
    return DinoSSLDataModule(config)


__all__ = ["DinoSSLDataModule", "build_dino_ssl_datamodule"]
