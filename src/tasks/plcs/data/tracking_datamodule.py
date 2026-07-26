"""DataModules for pre-generated and chunked multi-person PLCS data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from src.tasks.base.data.canonical_tracking import validate_lifecycle_capacity
from src.tasks.base.data.chunked_datamodule import BaseChunkedDataModule
from src.tasks.base.data.datamodule import SceneDirectoryDataModule
from src.tasks.plcs.data.chunk_manager import ChunkManager
from src.tasks.plcs.data.tracking_dataset import (
    PLCSTrackingDataset,
    collate_plcs_tracking_batch,
)


class PLCSTrackingDataModule(SceneDirectoryDataModule):
    """Read fixed train/val/test multi-person scenes from disk."""

    default_scene_dir = "data/plcs/multi_object"
    default_batch_size = 8

    def _build_collate_fn(self) -> Any:
        return collate_plcs_tracking_batch

    def _build_dataset(
        self, scene_dir: Path, split_file: str, augment: bool
    ) -> Dataset:
        return PLCSTrackingDataset(
            scene_dir=scene_dir,
            split_file=split_file,
            config=self.config,
            augment=augment,
        )

    def _dataset_name(self) -> str:
        return "plcs"


class ChunkedPLCSTrackingDataModule(BaseChunkedDataModule, PLCSTrackingDataModule):
    """Generate only train scenes on the fly while keeping val/test fixed."""

    def _default_chunks_dir(self) -> str:
        return "data/plcs/multi_object_chunks"

    def _build_chunk_manager(self) -> ChunkManager:
        generation_cfg: Any = self.config.get("generation")
        if generation_cfg is None or str(generation_cfg.get("mode")) != "multi_object":
            raise ValueError(
                "Chunked PLCS tracking requires generation.mode='multi_object'."
            )
        timeline_cfg: Any = generation_cfg.get("timeline")
        if timeline_cfg is None:
            raise ValueError("Chunked PLCS tracking requires generation.timeline.")
        validate_lifecycle_capacity(
            timeline_config=timeline_cfg,
            data_config=self.config.get("data", {}),
            num_queries=int(self.config.get("model", {}).get("num_queries")),
        )
        return ChunkManager(
            chunks_dir=self.chunks_dir,
            config=self.config,
            scenes_per_chunk=self.scenes_per_chunk,
            epochs_per_chunk=self.epochs_per_chunk,
            prefetch_chunks=self.prefetch_chunks,
            generator_device=self.generator_device,
            generation_workers=self.generation_workers,
        )


__all__ = ["PLCSTrackingDataModule", "ChunkedPLCSTrackingDataModule"]
