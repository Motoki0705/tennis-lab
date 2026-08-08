"""DataModule for canonical fixed-path multi-ball BLCS data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from src.tasks.base.data.datamodule import SceneDirectoryDataModule
from src.tasks.blcs.data.tracking_dataset import (
    BLCSTrackingDataset,
    collate_blcs_tracking_batch,
)


class BLCSTrackingDataModule(SceneDirectoryDataModule):
    """Read fixed train/val/test multi-ball scenes from disk."""

    def _build_collate_fn(self) -> Any:
        return collate_blcs_tracking_batch

    def _build_dataset(
        self, scene_dir: Path, split_file: str, augment: bool
    ) -> Dataset:
        return BLCSTrackingDataset(
            scene_dir=scene_dir,
            split_file=split_file,
            config=self.config,
            augment=augment,
        )

    def _dataset_name(self) -> str:
        return "blcs"


__all__ = ["BLCSTrackingDataModule"]
