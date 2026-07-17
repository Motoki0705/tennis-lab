"""DataModules for pre-generated and chunked multi-ball BLCS data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from src.tasks.base.data.chunked_datamodule import BaseChunkedDataModule
from src.tasks.base.data.datamodule import SceneDirectoryDataModule
from src.tasks.base.data.scene_chunk_manager import SceneChunkManager
from src.tasks.blcs.data.tracking_dataset import (
    BLCSTrackingDataset,
    collate_blcs_tracking_batch,
)
from src.tasks.blcs.generate_dataset.config import build_generator_config
from src.tasks.blcs.generate_dataset.io.dataset_io import BLCSDatasetWriter
from src.tasks.blcs.generate_dataset.multi_object_scene_generator import (
    MultiBallSceneGenerator,
)
from src.tasks.blcs.generate_dataset.scene_generator import BLCSSceneGenerator


class BLCSTrackingDataModule(SceneDirectoryDataModule):
    """Read fixed train/val/test multi-ball scenes from disk."""

    default_scene_dir = "data/blcs/multi_object"
    default_batch_size = 8

    def _build_collate_fn(self) -> Any:
        return collate_blcs_tracking_batch

    def _build_dataset(
        self, scene_dir: Path, split_file: str, augment: bool
    ) -> Dataset:
        del augment
        return BLCSTrackingDataset(scene_dir=scene_dir, split_file=split_file)

    def _dataset_name(self) -> str:
        return "blcs"


class ChunkedBLCSTrackingDataModule(
    BaseChunkedDataModule, BLCSTrackingDataModule
):
    """Generate only train scenes on the fly while keeping val/test fixed."""

    def _default_chunks_dir(self) -> str:
        return "data/blcs/multi_object_chunks"

    def _build_chunk_manager(self) -> SceneChunkManager:
        generation_cfg: Any = self.config.get("generation")
        if generation_cfg is None or str(generation_cfg.get("mode")) != "multi_object":
            raise ValueError(
                "Chunked BLCS tracking requires generation.mode='multi_object'."
            )
        generator = MultiBallSceneGenerator(
            BLCSSceneGenerator(
                build_generator_config(self.config), device=self.generator_device
            ),
            min_balls=int(generation_cfg.min_balls),
            max_balls=int(generation_cfg.max_balls),
        )
        return SceneChunkManager(
            chunks_dir=self.chunks_dir,
            writer_factory=BLCSDatasetWriter,
            scene_factory=generator.generate_scene,
            scenes_per_chunk=self.scenes_per_chunk,
            epochs_per_chunk=self.epochs_per_chunk,
            prefetch_chunks=self.prefetch_chunks,
        )


__all__ = ["BLCSTrackingDataModule", "ChunkedBLCSTrackingDataModule"]
