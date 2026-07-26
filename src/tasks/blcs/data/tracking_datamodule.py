"""DataModules for pre-generated and chunked multi-ball BLCS data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from src.tasks.base.data.canonical_tracking import validate_lifecycle_capacity
from src.tasks.base.data.chunked_datamodule import BaseChunkedDataModule
from src.tasks.base.data.datamodule import SceneDirectoryDataModule
from src.tasks.blcs.data.chunk_manager import ChunkManager
from src.tasks.blcs.data.tracking_dataset import (
    BLCSTrackingDataset,
    collate_blcs_tracking_batch,
)
from src.tasks.blcs.generate_dataset.config import build_generator_config


class BLCSTrackingDataModule(SceneDirectoryDataModule):
    """Read fixed train/val/test multi-ball scenes from disk."""

    default_scene_dir = "data/blcs/multi_object"
    default_batch_size = 8

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


class ChunkedBLCSTrackingDataModule(BaseChunkedDataModule, BLCSTrackingDataModule):
    """Generate only train scenes on the fly while keeping val/test fixed."""

    def _default_chunks_dir(self) -> str:
        return "data/blcs/multi_object_chunks"

    def _build_chunk_manager(self) -> ChunkManager:
        generation_cfg: Any = self.config.get("generation")
        if generation_cfg is None or str(generation_cfg.get("mode")) != "multi_object":
            raise ValueError(
                "Chunked BLCS tracking requires generation.mode='multi_object'."
            )
        timeline_cfg: Any = generation_cfg.get("timeline")
        if timeline_cfg is None:
            raise ValueError("Chunked BLCS tracking requires generation.timeline.")
        validate_lifecycle_capacity(
            timeline_config=timeline_cfg,
            data_config=self.config.get("data", {}),
            num_queries=int(self.config.get("model", {}).get("num_queries")),
        )
        return ChunkManager(
            chunks_dir=self.chunks_dir,
            generator_config=build_generator_config(self.config),
            scenes_per_chunk=self.scenes_per_chunk,
            epochs_per_chunk=self.epochs_per_chunk,
            prefetch_chunks=self.prefetch_chunks,
            generator_device=self.generator_device,
            generation_workers=self.generation_workers,
            multi_object=True,
            timeline_config=timeline_cfg,
        )


__all__ = ["BLCSTrackingDataModule", "ChunkedBLCSTrackingDataModule"]
