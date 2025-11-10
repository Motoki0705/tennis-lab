"""LightningDataModule that wires the DanceTrack dataset and collate."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.datasets.collate_tracking import SceneBatch, collate_tracking
from src.datasets.dancetrack import DancetrackDataset


class DancetrackDataModule(LightningDataModule):
    """DataModule that provides train/val DataLoaders for DanceTrack."""

    def __init__(
        self,
        dataset_cfg: DictConfig | Mapping[str, Any] | None,
        debug_cfg: DictConfig | Mapping[str, Any] | None,
    ) -> None:
        super().__init__()
        self.dataset_cfg = _to_dict(dataset_cfg)
        self.debug_cfg = _to_dict(debug_cfg)
        self.train_dataset: DancetrackDataset | None = None
        self.val_dataset: DancetrackDataset | None = None
        seed = self.debug_cfg.get("seed") or self.dataset_cfg.get("seed")
        self._generator = torch.Generator()
        if seed is None:
            self._generator = None
        else:
            self._generator.manual_seed(int(seed))

    def setup(self, stage: str | None = None) -> None:
        """Instantiate datasets for the requested Lightning stage."""
        if stage in (None, "fit"):
            self.train_dataset = self._build_dataset("train")
            self.val_dataset = self._build_dataset("val")
        elif stage in ("validate", "test"):
            if self.val_dataset is None:
                self.val_dataset = self._build_dataset("val")

    def _build_dataset(self, split: str) -> DancetrackDataset:
        return DancetrackDataset(self.dataset_cfg, split=split, debug=self.debug_cfg)

    def train_dataloader(self) -> DataLoader[SceneBatch]:
        """Return the training DataLoader configured via YAML."""
        dataset = self._require_dataset(self.train_dataset, "train")
        loader_cfg = self.dataset_cfg.get("loader", {}).get("train", {})
        return self._build_loader(
            dataset, loader_cfg, shuffle=loader_cfg.get("shuffle", True)
        )

    def val_dataloader(self) -> DataLoader[SceneBatch]:
        """Return the validation DataLoader configured via YAML."""
        dataset = self._require_dataset(self.val_dataset, "val")
        loader_cfg = self.dataset_cfg.get("loader", {}).get("val", {})
        return self._build_loader(
            dataset, loader_cfg, shuffle=loader_cfg.get("shuffle", False)
        )

    def _build_loader(
        self, dataset: DancetrackDataset, loader_cfg: Mapping[str, Any], shuffle: bool
    ) -> DataLoader[SceneBatch]:
        batch_size = int(loader_cfg.get("batch_size", 1))
        num_workers = int(loader_cfg.get("num_workers", 0))
        persistent_workers = (
            bool(loader_cfg.get("persistent_workers", False)) and num_workers > 0
        )
        generator = self._generator if shuffle and self._generator is not None else None
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=bool(loader_cfg.get("pin_memory", False)),
            drop_last=bool(loader_cfg.get("drop_last", False)),
            persistent_workers=persistent_workers,
            generator=generator,
            collate_fn=collate_tracking,
        )

    @staticmethod
    def _require_dataset(
        dataset: DancetrackDataset | None, split: str
    ) -> DancetrackDataset:
        if dataset is None:
            msg = f"Dataset for split '{split}' has not been set up yet"
            raise RuntimeError(msg)
        return dataset


def _to_dict(cfg: DictConfig | Mapping[str, Any] | None) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    if isinstance(cfg, DictConfig):
        return dict(OmegaConf.to_container(cfg, resolve=True))
    return dict(cfg)
