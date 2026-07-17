"""DataModules for pre-generated and chunked multi-person PLCS data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from src.tasks.base.data.chunked_datamodule import BaseChunkedDataModule
from src.tasks.base.data.datamodule import SceneDirectoryDataModule
from src.tasks.base.data.scene_chunk_manager import SceneChunkManager
from src.tasks.plcs.data.tracking_dataset import (
    PLCSTrackingDataset,
    collate_plcs_tracking_batch,
)
from src.tasks.plcs.generate_dataset.io.dataset_io import PLCSDatasetWriter
from src.tasks.plcs.generate_dataset.multi_object_scene_generator import (
    MultiPersonSceneGenerator,
)
from src.tasks.plcs.generate_dataset.utils.parallel_runner import build_scene_generator


class PLCSTrackingDataModule(SceneDirectoryDataModule):
    """Read fixed train/val/test multi-person scenes from disk."""

    default_scene_dir = "data/plcs/multi_object"
    default_batch_size = 8

    def _build_collate_fn(self) -> Any:
        return collate_plcs_tracking_batch

    def _build_dataset(
        self, scene_dir: Path, split_file: str, augment: bool
    ) -> Dataset:
        del augment
        return PLCSTrackingDataset(scene_dir=scene_dir, split_file=split_file)

    def _dataset_name(self) -> str:
        return "plcs"


class ChunkedPLCSTrackingDataModule(
    BaseChunkedDataModule, PLCSTrackingDataModule
):
    """Generate only train scenes on the fly while keeping val/test fixed."""

    def _default_chunks_dir(self) -> str:
        return "data/plcs/multi_object_chunks"

    def _build_chunk_manager(self) -> SceneChunkManager:
        generation_cfg: Any = self.config.get("generation")
        if generation_cfg is None or str(generation_cfg.get("mode")) != "multi_object":
            raise ValueError(
                "Chunked PLCS tracking requires generation.mode='multi_object'."
            )
        generator = MultiPersonSceneGenerator(
            build_scene_generator(self.config, self.generator_device),
            min_persons=int(generation_cfg.min_persons),
            max_persons=int(generation_cfg.max_persons),
        )
        return SceneChunkManager(
            chunks_dir=self.chunks_dir,
            writer_factory=PLCSDatasetWriter,
            scene_factory=generator.generate_scene,
            scenes_per_chunk=self.scenes_per_chunk,
            epochs_per_chunk=self.epochs_per_chunk,
            prefetch_chunks=self.prefetch_chunks,
        )


__all__ = ["PLCSTrackingDataModule", "ChunkedPLCSTrackingDataModule"]
