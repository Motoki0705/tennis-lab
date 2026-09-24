"""Lightning data module for PLCS triangulation-residual training."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.tasks.plcs.configuration_contracts import ResidualConfig
from src.tasks.plcs.data.residual_dataset import (
    ResidualDataset,
    _initialize_residual_worker,
    audit_source_splits,
    collate_residual,
    list_scene_paths,
    load_clean_scene,
)
from src.utils.configuration import PathRole


class ResidualDataModule(pl.LightningDataModule):
    def __init__(self, config: ResidualConfig) -> None:
        super().__init__()
        self.config = config
        self.datasets: dict[str, ResidualDataset] = {}
        self.split_audit: dict[str, Any] = {}

    def setup(self, stage: str | None = None) -> None:
        if self.datasets:
            return
        root = self.config.runtime.resolver.resolve(
            PathRole.DATA, self.config.data.scene_dir
        )
        self.split_audit = audit_source_splits(root)
        for split in ("train", "val", "test"):
            paths = list_scene_paths(root, split)
            limit = getattr(self.config.data, f"{split}_limit")
            if limit:
                paths = paths[:limit]
            self.datasets[split] = ResidualDataset(
                paths, load_clean_scene, self.config, split
            )
        self.train_dataset, self.val_dataset, self.test_dataset = (
            self.datasets[key] for key in ("train", "val", "test")
        )

    def _loader(self, split: str) -> DataLoader[dict[str, Any]]:
        cfg = self.config.data
        return DataLoader(
            self.datasets[split],
            batch_size=cfg.batch_size,
            shuffle=split == "train",
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.num_workers > 0,
            multiprocessing_context="spawn" if cfg.num_workers > 0 else None,
            worker_init_fn=_initialize_residual_worker,
            collate_fn=collate_residual,
            generator=torch.Generator().manual_seed(self.config.runtime.run.seed),
        )

    def train_dataloader(self) -> DataLoader[dict[str, Any]]:
        return self._loader("train")

    def val_dataloader(self) -> DataLoader[dict[str, Any]]:
        return self._loader("val")

    def test_dataloader(self) -> DataLoader[dict[str, Any]]:
        return self._loader("test")
