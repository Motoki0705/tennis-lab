"""PyTorch Lightning DataModule for event detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.tasks.event_detection.data.dataset import BLCSRallyEventDataset
from src.tasks.event_detection.data.types import Event3DBatch, Event3DSample, EventUVBatch, EventUVSample
from src.utils.data.collate import collate_padded_batch
from src.utils.data.scene_batch_sampler import build_scene_sampler

if TYPE_CHECKING:
    from omegaconf import DictConfig


def collate_uv(batch: list[EventUVSample]) -> EventUVBatch:
    """Collate UV samples with padding."""
    return collate_padded_batch(
        batch,
        sequence_keys=["ball_uv", "ball_vis", "targets"],
        static_keys=["court_kp", "court_vis"],
        seq_len_key="seq_len",
        mask_key="ball_mask",
    )


def collate_3d(batch: list[Event3DSample]) -> Event3DBatch:
    """Collate 3D samples with padding."""
    return collate_padded_batch(
        batch,
        sequence_keys=["ball_pos_world", "targets"],
        static_keys=[],
        seq_len_key="seq_len",
        mask_key=None,
    )


@dataclass(frozen=True)
class DataConfig:
    """Resolved datamodule configuration."""

    scene_dir: Path
    train_split_file: str
    val_split_file: str
    batch_size: int
    num_workers: int
    input_type: Literal["uv", "3d"]
    pin_memory: bool
    scene_sampler: bool
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
            train_split_file=str((data_cfg.get("split", {}) or {}).get("train_file", "train.txt")),
            val_split_file=str((data_cfg.get("split", {}) or {}).get("val_file", "val.txt")),
            batch_size=int(data_cfg.get("batch_size", 16)),
            num_workers=int(data_cfg.get("num_workers", 4)),
            input_type=input_type,
            pin_memory=bool(data_cfg.get("pin_memory", torch.cuda.is_available())),
            scene_sampler=bool(data_cfg.get("scene_sampler", True)),
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
                "Run src.tasks.blcs.scripts.generate_dataset to create the dataset."
            )

        if stage in ("fit", None):
            self.train_dataset = BLCSRallyEventDataset(
                scene_dir=scene_dir,
                split_file=self._resolved.train_split_file,
                input_type=self._resolved.input_type,
                config=self.config,
                augment=False,
            )
            self.val_dataset = BLCSRallyEventDataset(
                scene_dir=scene_dir,
                split_file=self._resolved.val_split_file,
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
            self._resolved.batch_size,
            enabled=self._resolved.scene_sampler,
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
            self._resolved.batch_size,
            enabled=self._resolved.scene_sampler,
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
