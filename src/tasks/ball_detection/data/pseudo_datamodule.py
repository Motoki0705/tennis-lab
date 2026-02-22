"""Lightning datamodule for pseudo-labeled self-training."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.ball_detection.data.datamodule import collate_ball_sequences
from src.ball_detection.data.pseudo_dataset import PseudoBallDataset


class BallDetectionPseudoDataModule(pl.LightningDataModule):
    """Datamodule that loads pseudo labels for sequence self-training."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        data_cfg = config.get("data", {})
        self.pseudo_root_dir = data_cfg.get("pseudo_root_dir", "outputs/ball_detection/pseudo")
        self.batch_size = int(data_cfg.get("batch_size", 16))
        self.num_workers = int(data_cfg.get("num_workers", 4))
        self.pin_memory = bool(data_cfg.get("pin_memory", True))
        self.persistent_workers = bool(data_cfg.get("persistent_workers", self.num_workers > 0))
        self.prefetch_factor = data_cfg.get("prefetch_factor", 2)
        self.image_size_hw = (int(data_cfg.get("image_h", 288)), int(data_cfg.get("image_w", 512)))
        self.window_size = int(data_cfg.get("window_size", 16))
        self.window_stride = int(data_cfg.get("window_stride", max(1, self.window_size // 2)))
        self.min_window_size = int(data_cfg.get("min_window_size", 4))

        self.dataset: PseudoBallDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.dataset = PseudoBallDataset(
                self.pseudo_root_dir,
                self.image_size_hw,
                window_size=self.window_size,
                window_stride=self.window_stride,
                min_window_size=self.min_window_size,
            )

    def train_dataloader(self) -> DataLoader:
        if self.dataset is None:
            raise RuntimeError("pseudo dataset is not initialized")
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": True,
            "collate_fn": collate_ball_sequences,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = int(self.prefetch_factor)
        return DataLoader(
            self.dataset,
            **kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        if self.dataset is None:
            raise RuntimeError("pseudo dataset is not initialized")
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": False,
            "collate_fn": collate_ball_sequences,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = int(self.prefetch_factor)
        return DataLoader(
            self.dataset,
            **kwargs,
        )

    def test_dataloader(self) -> DataLoader:
        if self.dataset is None:
            raise RuntimeError("pseudo dataset is not initialized")
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": False,
            "collate_fn": collate_ball_sequences,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = int(self.prefetch_factor)
        return DataLoader(
            self.dataset,
            **kwargs,
        )
