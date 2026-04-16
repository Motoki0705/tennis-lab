"""PyTorch Lightning DataModule for BLCS."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.tasks.blcs.data.dataset import BallTrajectoryDataset, collate_and_adapt_blcs_batch

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
            if not train_split.exists():
                raise RuntimeError(f"Missing required split file: {train_split}")
            self.train_dataset = BallTrajectoryDataset(
                scene_dir=self.scene_dir,
                split_file="train.txt",
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
            if not test_split.exists():
                raise RuntimeError(f"Missing required split file: {test_split}")
            self.test_dataset = BallTrajectoryDataset(
                scene_dir=self.scene_dir,
                split_file="test.txt",
                config=self.config,
                augment=False,
            )

    def _build_loader(self, dataset: BallTrajectoryDataset, *, train: bool) -> DataLoader:
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
