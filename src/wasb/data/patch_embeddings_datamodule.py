"""Lightning DataModule for cached patch embeddings and heatmaps."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.wasb.data.patch_embeddings_dataset import PatchEmbeddingsDataset


class PatchEmbeddingsDataModule(pl.LightningDataModule):
    """DataModule wrapping PatchEmbeddingsDataset."""

    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        cfg = config or {}
        data_cfg = cfg.get("data", {})

        self.root_dir = Path(data_cfg.get("root_dir", "data/tennis"))
        self.embeddings_dir = data_cfg.get("embeddings_dir", None)
        self.heatmaps_dir = data_cfg.get("heatmaps_dir", None)
        self.train_matches: Sequence[str] = data_cfg.get("train_matches", [])
        self.val_matches: Sequence[str] = data_cfg.get("val_matches", [])
        self.test_matches: Sequence[str] = data_cfg.get("test_matches", [])
        self.include_embeddings = bool(data_cfg.get("include_embeddings", True))
        self.include_heatmaps = bool(data_cfg.get("include_heatmaps", True))
        self.frames_in = int(data_cfg.get("frames_in", 8))
        self.frames_out = int(data_cfg.get("frames_out", 1))
        self.step = int(data_cfg.get("step", 1))
        self.batch_size = int(data_cfg.get("batch_size", 4))
        self.num_workers = int(data_cfg.get("num_workers", 4))
        self.pin_memory = bool(data_cfg.get("pin_memory", True))
        self.shuffle = bool(data_cfg.get("shuffle", True))
        self.drop_last = bool(data_cfg.get("drop_last", False))
        self.balance_augments = bool(data_cfg.get("balance_augments", False))
        self.samples_per_epoch = data_cfg.get("samples_per_epoch", None)
        self.sampler_replacement = bool(data_cfg.get("sampler_replacement", True))

        self.train_dataset: PatchEmbeddingsDataset | None = None
        self.val_dataset: PatchEmbeddingsDataset | None = None
        self.test_dataset: PatchEmbeddingsDataset | None = None
        self.train_sampler: torch.utils.data.Sampler[int] | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = PatchEmbeddingsDataset(
                root_dir=self.root_dir,
                embeddings_dir=self.embeddings_dir,
                heatmaps_dir=self.heatmaps_dir,
                matches=self.train_matches,
                include_embeddings=self.include_embeddings,
                include_heatmaps=self.include_heatmaps,
                frames_in=self.frames_in,
                frames_out=self.frames_out,
                step=self.step,
            )
            self.val_dataset = PatchEmbeddingsDataset(
                root_dir=self.root_dir,
                embeddings_dir=self.embeddings_dir,
                heatmaps_dir=self.heatmaps_dir,
                matches=self.val_matches or self.train_matches,
                include_embeddings=self.include_embeddings,
                include_heatmaps=self.include_heatmaps,
                frames_in=self.frames_in,
                frames_out=self.frames_out,
                step=self.step,
            )
            self.train_sampler = self._build_sampler(self.train_dataset)

        if stage in (None, "test"):
            self.test_dataset = PatchEmbeddingsDataset(
                root_dir=self.root_dir,
                embeddings_dir=self.embeddings_dir,
                heatmaps_dir=self.heatmaps_dir,
                matches=self.test_matches or self.val_matches or self.train_matches,
                include_embeddings=self.include_embeddings,
                include_heatmaps=self.include_heatmaps,
                frames_in=self.frames_in,
                frames_out=self.frames_out,
                step=self.step,
            )

    def _build_sampler(
        self, dataset: PatchEmbeddingsDataset
    ) -> torch.utils.data.Sampler[int] | None:
        if not self.balance_augments:
            return None
        aug_indices = dataset.sample_aug_indices()
        counts: dict[int | None, int] = {}
        for aug in aug_indices:
            counts[aug] = counts.get(aug, 0) + 1
        weights = torch.tensor([1.0 / counts[aug] for aug in aug_indices], dtype=torch.double)
        num_samples = self.samples_per_epoch
        return torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=int(num_samples) if num_samples is not None else len(weights),
            replacement=self.sampler_replacement,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not initialized")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle if self.train_sampler is None else False,
            sampler=self.train_sampler,
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is not initialized")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is not initialized")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )
