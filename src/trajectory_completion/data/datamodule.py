"""PyTorch Lightning DataModule for trajectory completion."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.trajectory_completion.data.dataset import (
    BLCSUVTrajectoryCompletionDataset,
)
from src.common.data.scene_batch_sampler import (
    build_scene_sampler,
    resolve_scene_sampler_mode,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


def collate_uv_trajectories(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Pad variable-length UV trajectories to the maximum length in the batch."""
    B = len(batch)
    max_len = max(int(s["ball_uv_gt"].shape[0]) for s in batch)

    ball_uv_in = torch.zeros(B, max_len, 2)
    ball_uv_gt = torch.zeros(B, max_len, 2)
    ball_vis = torch.zeros(B, max_len)
    ball_mask = torch.zeros(B, max_len)
    court_kp = torch.zeros(B, 20, 2)
    court_vis = torch.zeros(B, 20)
    seq_len = torch.zeros(B, dtype=torch.long)

    for i, s in enumerate(batch):
        T = int(s["ball_uv_gt"].shape[0])
        ball_uv_in[i, :T] = s["ball_uv_in"]
        ball_uv_gt[i, :T] = s["ball_uv_gt"]
        ball_vis[i, :T] = s["ball_vis"]
        ball_mask[i, :T] = 1.0
        court_kp[i] = s["court_kp"]
        court_vis[i] = s["court_vis"]
        seq_len[i] = s["seq_len"].to(torch.long)

    return {
        "ball_uv_in": ball_uv_in,
        "ball_vis": ball_vis,
        "ball_uv_gt": ball_uv_gt,
        "ball_mask": ball_mask,
        "court_kp": court_kp,
        "court_vis": court_vis,
        "seq_len": seq_len,
    }


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
        self.scene_sampler_mode = resolve_scene_sampler_mode(data_cfg)
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
            batch_size=self.batch_size,
            mode=self.scene_sampler_mode,
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
            batch_size=self.batch_size,
            mode=self.scene_sampler_mode,
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

