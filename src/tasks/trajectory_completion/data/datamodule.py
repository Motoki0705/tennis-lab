"""PyTorch Lightning DataModule for trajectory completion."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytorch_lightning as pl
from torch import Tensor
from torch.utils.data import DataLoader

from src.tasks.trajectory_completion.data.dataset import (
    BLCSUVTrajectoryCompletionDataset,
)
from src.utils.data.collate import collate_padded_batch
from src.utils.data.scene_batch_sampler import build_scene_sampler

if TYPE_CHECKING:
    from omegaconf import DictConfig


def collate_uv_trajectories(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Pad variable-length UV trajectories to the maximum length in the batch."""
    return collate_padded_batch(
        batch,
        sequence_keys=["ball_uv", "ball_vis", "ball_uv_gt", "ball_gt_vis", "ball_in_frame_gt"],
        static_keys=["court_kp", "court_vis"],
        seq_len_key="seq_len",
        mask_key="ball_mask",
    )


class TrajectoryCompletionDataModule(pl.LightningDataModule):
    """DataModule for UV trajectory completion training."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        data_cfg = config.get("data", {}) or {}

        self.scene_dir = Path(str(data_cfg.get("scene_dir", "data/blcs")))
        split_cfg = data_cfg.get("split", {}) or {}
        self.train_file = str(split_cfg.get("train_file", "train.txt"))
        self.val_file = str(split_cfg.get("val_file", "val.txt"))

        self.batch_size = int(data_cfg.get("batch_size", 16))
        self.num_workers = int(data_cfg.get("num_workers", 4))
        self.pin_memory = bool(data_cfg.get("pin_memory", True))
        self.scene_sampler = bool(data_cfg.get("scene_sampler", True))
        self.scenes_per_batch = int(data_cfg.get("scenes_per_batch", 1))
        self.chunk_max_scenes = int(data_cfg.get("chunk_max_scenes", 64))

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str | None = None) -> None:
        if not self.scene_dir.exists():
            raise RuntimeError(f"scene_dir not found: {self.scene_dir}")

        if stage in ("fit", None):
            self.train_dataset = BLCSUVTrajectoryCompletionDataset(
                scene_dir=self.scene_dir,
                split_file=self.train_file,
                config=self.config,
                augment=True,
            )
            self.val_dataset = BLCSUVTrajectoryCompletionDataset(
                scene_dir=self.scene_dir,
                split_file=self.val_file,
                config=self.config,
                augment=False,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before train_dataloader().")
        batch_sampler = build_scene_sampler(
            self.train_dataset,
            self.batch_size,
            enabled=self.scene_sampler,
            scenes_per_batch=self.scenes_per_batch,
            chunk_max_scenes=self.chunk_max_scenes,
            drop_last=True,
            shuffle=True,
        )
        if batch_sampler is not None:
            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate_uv_trajectories,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=collate_uv_trajectories,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before val_dataloader().")
        batch_sampler = build_scene_sampler(
            self.val_dataset,
            self.batch_size,
            enabled=self.scene_sampler,
            scenes_per_batch=self.scenes_per_batch,
            chunk_max_scenes=self.chunk_max_scenes,
            drop_last=False,
            shuffle=False,
        )
        if batch_sampler is not None:
            return DataLoader(
                self.val_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate_uv_trajectories,
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=collate_uv_trajectories,
        )
