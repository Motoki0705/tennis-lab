"""PLCS adapter for the shared chunk manager implementation."""

from __future__ import annotations

import threading
from pathlib import Path

from omegaconf import DictConfig

from src.tasks.base.data.chunk_manager import (
    ChunkManager as BaseChunkManager,
)
from src.tasks.plcs.generate_dataset.io.dataset_io import PLCSDatasetWriter
from src.tasks.plcs.generate_dataset.utils.parallel_runner import (
    generate_parallel_scenes,
)
from src.tasks.plcs.utils import prepare_generation_config


class _PLCSChunkGenerator:
    def __init__(
        self,
        *,
        config: DictConfig,
        generator_device: str,
        generation_workers: int,
    ) -> None:
        self.config = config
        self.generator_device = generator_device
        self.generation_workers = generation_workers
        self._next_scene_index = 0

        if self.generator_device != "cpu":
            raise ValueError(
                "Parallel PLCS chunk generation requires data.generator_device=cpu "
                f"when data.chunk.generation_workers={self.generation_workers}"
            )

    def __call__(
        self,
        chunk_dir: Path,
        *,
        num_scenes: int,
        stop_event: threading.Event,
    ) -> None:
        writer = PLCSDatasetWriter(str(chunk_dir))
        start_index = self._allocate_scene_range(num_scenes)

        for scene_data in generate_parallel_scenes(
            config=self.config,
            device=self.generator_device,
            start_index=start_index,
            num_scenes=num_scenes,
            num_workers=self.generation_workers,
        ):
            if stop_event.is_set():
                break
            writer.save_scene(scene_data)

    def _allocate_scene_range(self, num_scenes: int) -> int:
        start_index = self._next_scene_index
        self._next_scene_index += num_scenes
        return start_index


class ChunkManager(BaseChunkManager):
    """PLCS-compatible wrapper around the shared chunk manager."""

    def __init__(
        self,
        *,
        chunks_dir: str | Path,
        config: DictConfig,
        scenes_per_chunk: int = 1000,
        epochs_per_chunk: int = 3,
        prefetch_chunks: int = 1,
        generator_device: str = "cpu",
        generation_workers: int = 1,
    ) -> None:
        self.generation_workers = generation_workers
        self.config = prepare_generation_config(config)
        self.generator_device = generator_device

        super().__init__(
            chunks_dir=chunks_dir,
            chunk_generator_factory=lambda: _PLCSChunkGenerator(
                config=self.config,
                generator_device=generator_device,
                generation_workers=self.generation_workers,
            ),
            scenes_per_chunk=scenes_per_chunk,
            epochs_per_chunk=epochs_per_chunk,
            prefetch_chunks=prefetch_chunks,
        )


__all__ = ["ChunkManager"]
