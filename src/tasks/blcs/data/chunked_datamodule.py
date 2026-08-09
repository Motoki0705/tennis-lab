"""PyTorch Lightning DataModule for chunked BLCS training.

The training set is backed by :class:`ChunkManager` which generates scene
chunks in a background thread.  Validation and test sets are fixed NPZ
datasets loaded from ``data/blcs/``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from src.tasks.base.data.chunked_datamodule import BaseChunkedDataModule
from src.tasks.blcs.data.chunk_manager import ChunkManager
from src.tasks.blcs.data.datamodule import BLCSDataModuleHooks

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from src.tasks.blcs.generate_dataset.scene_generator import GeneratorConfig


class ChunkedBLCSDataModule(BLCSDataModuleHooks, BaseChunkedDataModule):
    """DataModule that swaps train chunks generated in the background.

    Val/test splits are loaded once from the fixed ``scene_dir``.
    """

    def __init__(
        self,
        config: DictConfig,
        *,
        generator_config: GeneratorConfig,
        collate_fn: Callable[..., Any],
    ) -> None:
        self._collate_fn = collate_fn
        super().__init__(config)
        self.generator_config = generator_config

    def _build_chunk_manager(self) -> ChunkManager:
        config: Any = self.config
        return ChunkManager(
            chunks_dir=self.chunks_dir,
            generator_config=self.generator_config,
            scenes_per_chunk=self.scenes_per_chunk,
            epochs_per_chunk=self.epochs_per_chunk,
            prefetch_chunks=self.prefetch_chunks,
            generator_device=self.generator_device,
            generation_workers=self.generation_workers,
            generation_chunksize=int(config.data.chunk.generation_chunksize),
            generation_seed=int(config.run.seed),
            multi_object=False,
            timeline_config=None,
        )
