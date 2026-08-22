"""DataModules for pre-generated and chunked multi-ball BLCS data."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

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

if TYPE_CHECKING:
    from omegaconf import DictConfig


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


class ChunkedBLCSTrackingDataModule(BaseChunkedDataModule, BLCSTrackingDataModule):
    """Generate only train scenes on the fly while keeping val/test fixed."""

    def _build_chunk_manager(self) -> ChunkManager:
        config: Any = self.config
        generation_cfg: Any = config.generation
        if str(generation_cfg.mode) != "multi_object":
            raise ValueError(
                "Chunked BLCS tracking requires generation.mode='multi_object'."
            )
        timeline_cfg: Any = generation_cfg.timeline
        validate_lifecycle_capacity(
            timeline_config=timeline_cfg,
            data_config=config.data,
            num_queries=int(config.model.num_queries),
        )
        return ChunkManager(
            chunks_dir=self.chunks_dir,
            generator_config=build_generator_config(cast("DictConfig", self.config)),
            scenes_per_chunk=self.scenes_per_chunk,
            epochs_per_chunk=self.epochs_per_chunk,
            prefetch_chunks=self.prefetch_chunks,
            generator_device=self.generator_device,
            generation_workers=self.generation_workers,
            generation_chunksize=int(config.data.chunk.generation_chunksize),
            generation_seed=int(config.run.seed),
            multi_object=True,
            timeline_config=timeline_cfg,
            maximum_physics_attempts_per_object=int(
                generation_cfg.maximum_physics_attempts_per_object
            ),
        )


__all__ = ["BLCSTrackingDataModule", "ChunkedBLCSTrackingDataModule"]
