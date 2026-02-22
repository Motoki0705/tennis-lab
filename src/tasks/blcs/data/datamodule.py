"""PyTorch Lightning DataModule for BLCS."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.blcs.data.dataset import BallTrajectoryDataset, collate_and_adapt_blcs_batch
from src.common.data.scene_batch_sampler import build_scene_sampler, resolve_scene_sampler_mode

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BLCSDataModule(pl.LightningDataModule):
    """Lightning DataModule for unified BLCS single/multiview training."""

    def __init__(self, config: DictConfig | None = None) -> None:
        super().__init__()
        self.config = config or {}

        data_cfg = self.config.get("data", {})
        self.batch_size = int(data_cfg.get("batch_size", 32))
        self.num_workers = int(data_cfg.get("num_workers", 4))
        self.pin_memory = bool(data_cfg.get("pin_memory", True))
        self.scene_dir = Path(data_cfg.get("scene_dir", "data/blcs"))

        self.scene_sampler_mode = resolve_scene_sampler_mode(data_cfg)
        self.scenes_per_batch = int(data_cfg.get("scenes_per_batch", 1))
        self.chunk_max_scenes = int(data_cfg.get("chunk_max_scenes", 64))

        self.input_profile = str(self.config["model"]["io"]["input_profile"])
        if self.input_profile not in {"single", "multiview"}:
            raise ValueError(
                "Invalid model.io.input_profile="
                f"'{self.input_profile}'. Supported: ['single', 'multiview']"
            )
        self.collate_fn = partial(
            collate_and_adapt_blcs_batch,
            input_profile=self.input_profile,
        )

        self.train_dataset: BallTrajectoryDataset | None = None
        self.val_dataset: BallTrajectoryDataset | None = None
        self.test_dataset: BallTrajectoryDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if not self.scene_dir.exists():
            raise RuntimeError(
                f"Scene directory not found: {self.scene_dir}. "
                "Run blcs.scripts.generate_dataset to create the dataset."
            )

        if stage == "fit" or stage is None:
            train_split = self.scene_dir / "train.txt"
            if train_split.exists():
                self.train_dataset = BallTrajectoryDataset(
                    scene_dir=self.scene_dir,
                    split_file="train.txt",
                    config=self.config,
                    augment=True,
                )
            else:
                self.train_dataset = BallTrajectoryDataset(
                    scene_dir=self.scene_dir,
                    config=self.config,
                    augment=True,
                )

            val_split = self.scene_dir / "val.txt"
            if val_split.exists():
                self.val_dataset = BallTrajectoryDataset(
                    scene_dir=self.scene_dir,
                    split_file="val.txt",
                    config=self.config,
                    augment=False,
                )
            else:
                self.val_dataset = self.train_dataset

        if stage == "test" or stage is None:
            test_split = self.scene_dir / "test.txt"
            if test_split.exists():
                self.test_dataset = BallTrajectoryDataset(
                    scene_dir=self.scene_dir,
                    split_file="test.txt",
                    config=self.config,
                    augment=False,
                )
            else:
                self.test_dataset = BallTrajectoryDataset(
                    scene_dir=self.scene_dir,
                    config=self.config,
                    augment=False,
                )

    def _build_loader(self, dataset: BallTrajectoryDataset, *, train: bool) -> DataLoader:
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
                collate_fn=self.collate_fn,
            )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=train,
            collate_fn=self.collate_fn,
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
