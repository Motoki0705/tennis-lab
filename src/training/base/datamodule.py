"""Shared LightningDataModule utilities for reusable training stacks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, Generic, TypeVar

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

DatasetT = TypeVar("DatasetT")
BatchT = TypeVar("BatchT")


class BaseDataModule(LightningDataModule, Generic[DatasetT, BatchT], ABC):
    """Convenience base that handles cfg conversion and DataLoader wiring."""

    def __init__(
        self,
        dataset_cfg: DictConfig | Mapping[str, Any] | None,
        debug_cfg: DictConfig | Mapping[str, Any] | None,
    ) -> None:
        super().__init__()
        self.dataset_cfg = self._to_dict(dataset_cfg)
        self.debug_cfg = self._to_dict(debug_cfg)
        self.train_dataset: DatasetT | None = None
        self.val_dataset: DatasetT | None = None
        self.test_dataset: DatasetT | None = None
        self._generator = self._build_generator()

    @abstractmethod
    def _build_dataset(self, split: str) -> DatasetT:
        """Create the dataset for the specified split."""

    def collate_fn(self) -> Callable[..., Any] | None:
        """Optionally override to provide a custom collate function."""
        return None

    def _build_generator(self) -> torch.Generator | None:
        seed = self.debug_cfg.get("seed") or self.dataset_cfg.get("seed")
        if seed is None:
            return None
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        return generator

    def _loader_cfg(self, phase: str) -> Mapping[str, Any]:
        loader_cfg = self.dataset_cfg.get("loader", {})
        if isinstance(loader_cfg, Mapping):
            cfg = loader_cfg.get(phase, {})
            return cfg if isinstance(cfg, Mapping) else {}
        return {}

    def _make_loader(
        self,
        dataset: DatasetT | None,
        split: str,
        phase: str,
        default_shuffle: bool,
    ) -> DataLoader[BatchT]:
        loader_cfg = self._loader_cfg(phase)
        shuffle = bool(loader_cfg.get("shuffle", default_shuffle))
        resolved_dataset = self._require_dataset(dataset, split)
        return self._build_loader(resolved_dataset, loader_cfg, shuffle)

    def _build_loader(
        self,
        dataset: DatasetT,
        loader_cfg: Mapping[str, Any],
        shuffle: bool,
    ) -> DataLoader[BatchT]:
        batch_size = int(loader_cfg.get("batch_size", 1))
        num_workers = int(loader_cfg.get("num_workers", 0))
        persistent_workers = bool(loader_cfg.get("persistent_workers", False))
        persistent_workers = persistent_workers and num_workers > 0
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
            collate_fn=self.collate_fn(),
        )

    @staticmethod
    def _require_dataset(dataset: DatasetT | None, split: str) -> DatasetT:
        if dataset is None:
            msg = f"Dataset for split '{split}' has not been set up yet"
            raise RuntimeError(msg)
        return dataset

    @staticmethod
    def _to_dict(cfg: DictConfig | Mapping[str, Any] | None) -> dict[str, Any]:
        if cfg is None:
            return {}
        if isinstance(cfg, dict):
            return cfg
        if isinstance(cfg, DictConfig):
            return dict(OmegaConf.to_container(cfg, resolve=True))
        return dict(cfg)
