"""PLCS adapter for the shared chunk manager implementation."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.tasks.base.data.chunk_manager import (
    ChunkGenerator,
    ChunkInfo,
    ChunkManager as BaseChunkManager,
    ChunkState,
)
from src.tasks.plcs.generate_dataset.io.dataset_io import PLCSDatasetWriter
from src.tasks.plcs.generate_dataset.utils.parallel_runner import (
    build_scene_generator,
    generate_parallel_scenes,
    generate_serial_scene,
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
        self._scene_generator: Any | None = None

        if self.generation_workers > 0 and self.generator_device != "cpu":
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

        if self.generation_workers > 0:
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
        else:
            scene_generator = self._get_scene_generator()
            for scene_index in range(start_index, start_index + num_scenes):
                if stop_event.is_set():
                    break
                writer.save_scene(
                    generate_serial_scene(
                        scene_generator,
                        scene_index,
                    )
                )

    def _get_scene_generator(self) -> Any:
        if self._scene_generator is None:
            self._scene_generator = build_scene_generator(self.config, self.generator_device)
        return self._scene_generator

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
        generation_workers: int = 0,
    ) -> None:
        self.config = prepare_generation_config(config)
        self.generator_device = generator_device
        self.generation_workers = generation_workers

        super().__init__(
            chunks_dir=chunks_dir,
            chunk_generator_factory=lambda: _PLCSChunkGenerator(
                config=self.config,
                generator_device=generator_device,
                generation_workers=generation_workers,
            ),
            scenes_per_chunk=scenes_per_chunk,
            epochs_per_chunk=epochs_per_chunk,
            prefetch_chunks=prefetch_chunks,
        )


__all__ = ["ChunkGenerator", "ChunkInfo", "ChunkManager", "ChunkState"]
