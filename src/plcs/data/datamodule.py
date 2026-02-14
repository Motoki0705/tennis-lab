"""Unified PyTorch Lightning DataModule for PLCS."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from src.common.data.scene_batch_sampler import build_scene_sampler, resolve_scene_sampler_mode
from src.plcs.data.dataset import SceneDataset, collate_plcs_batch

if TYPE_CHECKING:
    from omegaconf import DictConfig


class PLCSDataModule(pl.LightningDataModule):
    """Lightning DataModule for unified PLCS frame/sequence/multiview training."""

    def __init__(self, config: DictConfig | None = None) -> None:
        super().__init__()
        self.config = config or {}

        data_cfg = self.config.get("data", {})
        self.batch_size = int(data_cfg.get("batch_size", 64))
        self.num_workers = int(data_cfg.get("num_workers", 4))
        self.pin_memory = bool(data_cfg.get("pin_memory", True))
        self.scene_dir = Path(data_cfg.get("scene_dir", "data/plcs"))

        self.val_split = float(data_cfg.get("val_split", 0.1))
        self.test_split = float(data_cfg.get("test_split", 0.1))

        self.scene_sampler_mode = resolve_scene_sampler_mode(data_cfg)
        self.scenes_per_batch = int(data_cfg.get("scenes_per_batch", 1))
        self.chunk_max_scenes = int(data_cfg.get("chunk_max_scenes", 64))

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None) -> None:
        if not self.scene_dir.exists():
            raise RuntimeError(
                f"Scene directory not found: {self.scene_dir}. "
                "Run plcs.scripts.generate_dataset to create the dataset."
            )

        full_dataset = SceneDataset(
            scene_dir=self.scene_dir,
            config=self.config,
            augment=True,
        )

        total_len = len(full_dataset)
        val_len = int(total_len * self.val_split)
        test_len = int(total_len * self.test_split)
        train_len = total_len - val_len - test_len
        if train_len <= 0:
            raise ValueError(
                f"Invalid split sizes: train={train_len}, val={val_len}, test={test_len}."
            )

        train_ds, val_ds, test_ds = random_split(
            full_dataset,
            [train_len, val_len, test_len],
        )

        if stage == "fit" or stage is None:
            self.train_dataset = train_ds
            self.val_dataset = val_ds

        if stage == "test" or stage is None:
            self.test_dataset = test_ds

    def _build_loader(self, dataset, *, train: bool) -> DataLoader:
        batch_sampler = build_scene_sampler(
            dataset,
            batch_size=self.batch_size,
            mode=self.scene_sampler_mode,
            scenes_per_batch=self.scenes_per_batch,
            chunk_max_scenes=self.chunk_max_scenes,
            drop_last=train,
            shuffle=train,
        )
        if batch_sampler is not None:
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate_plcs_batch,
            )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=train,
            collate_fn=collate_plcs_batch,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before train_dataloader()")
        return self._build_loader(self.train_dataset, train=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit') before val_dataloader()")
        return self._build_loader(self.val_dataset, train=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Call setup('test') before test_dataloader()")
        return self._build_loader(self.test_dataset, train=False)
