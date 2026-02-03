"""PyTorch Lightning DataModule for event detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.evnet_detection.data.dataset import BLCSRallyEventDataset
from src.evnet_detection.data.types import Event3DBatch, Event3DSample, EventUVBatch, EventUVSample
from src.common.data.scene_batch_sampler import (
    build_scene_sampler,
    resolve_scene_sampler_mode,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


def collate_uv(batch: list[EventUVSample]) -> EventUVBatch:
    """Collate UV samples with padding."""
    B = len(batch)
    max_len = max(int(s["seq_len"]) for s in batch)
    E = int(batch[0]["targets"].shape[-1])

    ball_uv = torch.zeros(B, max_len, 2)
    ball_vis = torch.zeros(B, max_len)
    ball_mask = torch.zeros(B, max_len)
    targets = torch.zeros(B, max_len, E)
    court_kp = torch.zeros(B, 20, 2)
    court_vis = torch.zeros(B, 20)
    seq_len = torch.zeros(B, dtype=torch.long)

    for i, s in enumerate(batch):
        L = int(s["seq_len"])
        ball_uv[i, :L] = s["ball_uv"][:L]
        ball_vis[i, :L] = s["ball_vis"][:L]
        ball_mask[i, :L] = 1.0
        targets[i, :L] = s["targets"][:L]
        court_kp[i] = s["court_kp"]
        court_vis[i] = s["court_vis"]
        seq_len[i] = L

    return {
        "ball_uv": ball_uv,
        "ball_vis": ball_vis,
        "ball_mask": ball_mask,
        "court_kp": court_kp,
        "court_vis": court_vis,
        "targets": targets,
        "seq_len": seq_len,
    }


def collate_3d(batch: list[Event3DSample]) -> Event3DBatch:
    """Collate 3D samples with padding."""
    B = len(batch)
    max_len = max(int(s["seq_len"]) for s in batch)
    E = int(batch[0]["targets"].shape[-1])

    ball_pos = torch.zeros(B, max_len, 3)
    targets = torch.zeros(B, max_len, E)
    seq_len = torch.zeros(B, dtype=torch.long)

    for i, s in enumerate(batch):
        L = int(s["seq_len"])
        ball_pos[i, :L] = s["ball_pos_world"][:L]
        targets[i, :L] = s["targets"][:L]
        seq_len[i] = L

    return {"ball_pos_world": ball_pos, "targets": targets, "seq_len": seq_len}


@dataclass(frozen=True)
class DataConfig:
    """Resolved datamodule configuration."""

    scene_dir: Path
    batch_size: int
    num_workers: int
    input_type: Literal["uv", "3d"]
    pin_memory: bool
    scene_sampler_mode: str
    scenes_per_batch: int
    chunk_max_scenes: int


class EventDetectionDataModule(pl.LightningDataModule):
    """Lightning DataModule for event detection training."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config

        data_cfg = self.config.get("data", {}) or {}
        model_cfg = self.config.get("model", {}) or {}
        name = str(model_cfg.get("name", "uv_transformer"))
        input_type: Literal["uv", "3d"] = "3d" if "traj3d" in name else "uv"

        self._resolved = DataConfig(
            scene_dir=Path(str(data_cfg.get("scene_dir", "data/blcs"))),
            batch_size=int(data_cfg.get("batch_size", 16)),
            num_workers=int(data_cfg.get("num_workers", 4)),
            input_type=input_type,
            pin_memory=bool(data_cfg.get("pin_memory", torch.cuda.is_available())),
            scene_sampler_mode=resolve_scene_sampler_mode(data_cfg),
            scenes_per_batch=int(data_cfg.get("scenes_per_batch", 1)),
            chunk_max_scenes=int(data_cfg.get("chunk_max_scenes", 64)),
        )

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str | None = None) -> None:
        scene_dir = self._resolved.scene_dir
        if not scene_dir.exists():
            raise RuntimeError(
                f"Scene directory not found: {scene_dir}. "
                "Run src.blcs.scripts.generate_dataset to create the dataset."
            )

        if stage in ("fit", None):
            self.train_dataset = BLCSRallyEventDataset(
                scene_dir=scene_dir,
                split="train",
                input_type=self._resolved.input_type,
                config=self.config,
                augment=False,
            )
            self.val_dataset = BLCSRallyEventDataset(
                scene_dir=scene_dir,
                split="val",
                input_type=self._resolved.input_type,
                config=self.config,
                augment=False,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before train_dataloader().")
        collate = collate_3d if self._resolved.input_type == "3d" else collate_uv
        batch_sampler = build_scene_sampler(
            self.train_dataset,
            batch_size=self._resolved.batch_size,
            mode=self._resolved.scene_sampler_mode,
            scenes_per_batch=self._resolved.scenes_per_batch,
            chunk_max_scenes=self._resolved.chunk_max_scenes,
            drop_last=True,
            shuffle=True,
        )
        if batch_sampler is not None:
            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                num_workers=self._resolved.num_workers,
                pin_memory=self._resolved.pin_memory,
                collate_fn=collate,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self._resolved.batch_size,
            shuffle=True,
            num_workers=self._resolved.num_workers,
            pin_memory=self._resolved.pin_memory,
            drop_last=True,
            collate_fn=collate,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit') before val_dataloader().")
        collate = collate_3d if self._resolved.input_type == "3d" else collate_uv
        batch_sampler = build_scene_sampler(
            self.val_dataset,
            batch_size=self._resolved.batch_size,
            mode=self._resolved.scene_sampler_mode,
            scenes_per_batch=self._resolved.scenes_per_batch,
            chunk_max_scenes=self._resolved.chunk_max_scenes,
            drop_last=False,
            shuffle=False,
        )
        if batch_sampler is not None:
            return DataLoader(
                self.val_dataset,
                batch_sampler=batch_sampler,
                num_workers=self._resolved.num_workers,
                pin_memory=self._resolved.pin_memory,
                collate_fn=collate,
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self._resolved.batch_size,
            shuffle=False,
            num_workers=self._resolved.num_workers,
            pin_memory=self._resolved.pin_memory,
            collate_fn=collate,
        )

