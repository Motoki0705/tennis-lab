"""PyTorch Lightning DataModule for ball multi-task training."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.ball_multitask.data.dataset import BallMultitaskDataset
from src.common.data.scene_batch_sampler import build_scene_sampler, resolve_scene_sampler_mode

if TYPE_CHECKING:
    from omegaconf import DictConfig


def collate_multitask(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Pad variable-length sequences for multi-task batches."""
    B = len(batch)
    max_len = max(int(s["ball_uv_gt"].shape[0]) for s in batch)
    E = int(batch[0]["event_targets"].shape[-1])

    ball_uv_in = torch.zeros(B, max_len, 2)
    ball_uv_gt = torch.zeros(B, max_len, 2)
    ball_vis = torch.zeros(B, max_len)
    ball_in_frame_gt = torch.zeros(B, max_len)
    ball_mask = torch.zeros(B, max_len)
    court_kp = torch.zeros(B, 20, 2)
    court_vis = torch.zeros(B, 20)
    position_3d = torch.zeros(B, max_len, 3)
    ball_pos_world = torch.zeros(B, max_len, 3)
    event_targets = torch.zeros(B, max_len, E)
    seq_len = torch.zeros(B, dtype=torch.long)

    for i, s in enumerate(batch):
        T = int(s["ball_uv_gt"].shape[0])
        L = int(s["seq_len"].item())
        ball_uv_in[i, :T] = s["ball_uv_in"]
        ball_uv_gt[i, :T] = s["ball_uv_gt"]
        ball_vis[i, :T] = s["ball_vis"]
        ball_in_frame_gt[i, :T] = s["ball_in_frame_gt"]
        ball_mask[i, :L] = 1.0
        court_kp[i] = s["court_kp"]
        court_vis[i] = s["court_vis"]
        position_3d[i, :T] = s["position_3d"]
        ball_pos_world[i, :T] = s["ball_pos_world"]
        event_targets[i, :T] = s["event_targets"]
        seq_len[i] = L

    return {
        "ball_uv_in": ball_uv_in,
        "ball_uv_gt": ball_uv_gt,
        "ball_vis": ball_vis,
        "ball_in_frame_gt": ball_in_frame_gt,
        "ball_mask": ball_mask,
        "court_kp": court_kp,
        "court_vis": court_vis,
        "position_3d": position_3d,
        "ball_pos_world": ball_pos_world,
        "event_targets": event_targets,
        "seq_len": seq_len,
    }


class BallMultitaskDataModule(pl.LightningDataModule):
    """DataModule for unified multi-task training."""

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
        self.pin_memory = bool(data_cfg.get("pin_memory", torch.cuda.is_available()))
        self.scene_sampler_mode = resolve_scene_sampler_mode(data_cfg)
        self.scenes_per_batch = int(data_cfg.get("scenes_per_batch", 1))
        self.chunk_max_scenes = int(data_cfg.get("chunk_max_scenes", 64))

        self.train_dataset: BallMultitaskDataset | None = None
        self.val_dataset: BallMultitaskDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if not self.scene_dir.exists():
            raise RuntimeError(
                f"scene_dir not found: {self.scene_dir}. Run src.blcs.scripts.generate_dataset."
            )

        if stage in ("fit", None):
            self.train_dataset = BallMultitaskDataset(
                scene_dir=self.scene_dir,
                split_file=self.train_file,
                config=self.config,
                augment=True,
            )
            self.val_dataset = BallMultitaskDataset(
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
                collate_fn=collate_multitask,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=collate_multitask,
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
                collate_fn=collate_multitask,
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=collate_multitask,
        )
