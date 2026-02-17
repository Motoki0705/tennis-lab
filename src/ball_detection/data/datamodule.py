"""Lightning datamodule for supervised ball detection training."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
from torch import Tensor
from torch.utils.data import DataLoader

from src.ball_detection.data.labeled_dataset import LabeledBallDataset
from src.common.dataset.collate import collate_padded_batch


def collate_ball_sequences(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Pad variable-length frame sequences and build a valid-frame mask."""
    return collate_padded_batch(
        batch,
        sequence_keys=["frames", "target_xy", "target_vis", "target_weight"],
        static_keys=[],
        seq_len_key="seq_len",
        mask_key="frame_mask",
    )


class BallDetectionDataModule(pl.LightningDataModule):
    """Datamodule for labeled supervised sequence training."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        data_cfg = config.get("data", {})
        self.root_dir = data_cfg.get("root_dir", "data/tennis")
        self.train_games = list(data_cfg.get("train_games", []))
        self.val_games = list(data_cfg.get("val_games", []))
        self.batch_size = int(data_cfg.get("batch_size", 16))
        self.num_workers = int(data_cfg.get("num_workers", 4))
        self.pin_memory = bool(data_cfg.get("pin_memory", True))
        self.persistent_workers = bool(data_cfg.get("persistent_workers", self.num_workers > 0))
        self.prefetch_factor = data_cfg.get("prefetch_factor", 2)
        self.image_size_hw = (int(data_cfg.get("image_h", 288)), int(data_cfg.get("image_w", 512)))
        self.window_size = int(data_cfg.get("window_size", data_cfg.get("max_seq_len", 16)))
        self.window_stride = int(data_cfg.get("window_stride", max(1, self.window_size // 2)))
        self.min_window_size = int(data_cfg.get("min_window_size", 4))

        self.train_dataset: LabeledBallDataset | None = None
        self.val_dataset: LabeledBallDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = LabeledBallDataset(
                self.root_dir,
                self.train_games,
                self.image_size_hw,
                window_size=self.window_size,
                window_stride=self.window_stride,
                min_window_size=self.min_window_size,
            )
            self.val_dataset = LabeledBallDataset(
                self.root_dir,
                self.val_games or self.train_games,
                self.image_size_hw,
                window_size=self.window_size,
                window_stride=self.window_stride,
                min_window_size=self.min_window_size,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not initialized")
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
            self.train_dataset,
            **kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is not initialized")
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
            self.val_dataset,
            **kwargs,
        )
