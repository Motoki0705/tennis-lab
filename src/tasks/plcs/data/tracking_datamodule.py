"""DataModule for canonical fixed-path multi-person PLCS data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from src.tasks.base.data.datamodule import SceneDirectoryDataModule
from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.data.tracking_dataset import (
    PLCSTrackingDataset,
    collate_plcs_tracking_batch,
)


class PLCSTrackingDataModule(SceneDirectoryDataModule):
    """Read fixed train/val/test multi-person scenes from disk."""

    def __init__(self, config: object) -> None:
        self.plcs_runtime = PLCSTrainingConfig.from_config(config)
        super().__init__(config)

    def _build_collate_fn(self) -> Any:
        return collate_plcs_tracking_batch

    def _build_dataset(
        self, scene_dir: Path, split_file: str, augment: bool
    ) -> Dataset:
        return PLCSTrackingDataset(
            scene_dir=scene_dir,
            split_file=split_file,
            config=self.plcs_runtime.raw,
            augment=augment,
        )

    def _dataset_name(self) -> str:
        return "plcs"


__all__ = ["PLCSTrackingDataModule"]
