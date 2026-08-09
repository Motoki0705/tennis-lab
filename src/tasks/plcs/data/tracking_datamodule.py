"""DataModules for pre-generated and chunked multi-person PLCS data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from src.tasks.base.data.canonical_tracking import validate_lifecycle_capacity
from src.tasks.base.data.chunked_datamodule import BaseChunkedDataModule
from src.tasks.base.data.datamodule import SceneDirectoryDataModule
from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.data.chunk_manager import PLCSChunkManager
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


class ChunkedPLCSTrackingDataModule(BaseChunkedDataModule, PLCSTrackingDataModule):
    """Generate only train scenes on the fly while keeping val/test fixed."""

    def _build_chunk_manager(self) -> PLCSChunkManager:
        generation_cfg = self.plcs_runtime.raw.generation
        if str(generation_cfg.mode) != "multi_object":
            raise ValueError(
                "Chunked PLCS tracking requires generation.mode='multi_object'."
            )
        validate_lifecycle_capacity(
            timeline_config=generation_cfg.timeline,
            data_config=self.plcs_runtime.raw.data,
            num_queries=self.plcs_runtime.model.integer("num_queries"),
        )
        return PLCSChunkManager(
            chunks_dir=self.chunks_dir,
            config=self.plcs_runtime.raw,
            scenes_per_chunk=self.scenes_per_chunk,
            epochs_per_chunk=self.epochs_per_chunk,
            prefetch_chunks=self.prefetch_chunks,
            generator_device=self.generator_device,
            generation_workers=self.generation_workers,
        )


__all__ = ["PLCSTrackingDataModule", "ChunkedPLCSTrackingDataModule"]
