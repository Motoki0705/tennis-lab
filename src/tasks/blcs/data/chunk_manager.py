"""BLCS adapter for the shared chunk manager implementation."""

from __future__ import annotations

import threading
from pathlib import Path

from src.tasks.base.data.chunk_manager import (
    ChunkGenerator,
    ChunkInfo,
    ChunkManager as BaseChunkManager,
    ChunkState,
)
from src.tasks.blcs.generate_dataset.io.dataset_io import BLCSDatasetWriter
from src.tasks.blcs.generate_dataset.scene_generator import (
    BLCSSceneGenerator,
    GeneratorConfig,
)
from src.tasks.blcs.generate_dataset.utils.parallel_runner import (
    generate_parallel_scenes,
)


class _BLCSChunkGenerator:
    def __init__(
        self,
        *,
        generator_config: GeneratorConfig,
        generator_device: str,
        generation_workers: int,
    ) -> None:
        self.generator_config = generator_config
        self.generator_device = generator_device
        self.generation_workers = generation_workers
        self._generator: BLCSSceneGenerator | None = None

    def __call__(
        self,
        chunk_dir: Path,
        *,
        num_scenes: int,
        stop_event: threading.Event,
    ) -> None:
        writer = BLCSDatasetWriter(str(chunk_dir))
        if self.generation_workers > 0:
            for scene_data in generate_parallel_scenes(
                generator_config=self.generator_config,
                device=self.generator_device,
                num_scenes=num_scenes,
                num_workers=self.generation_workers,
            ):
                if stop_event.is_set():
                    break
                writer.save_scene(scene_data)
            return

        for scene_data in self._get_generator().generate(num_scenes):
            if stop_event.is_set():
                break
            writer.save_scene(scene_data)

    def _get_generator(self) -> BLCSSceneGenerator:
        if self._generator is None:
            self._generator = BLCSSceneGenerator(
                config=self.generator_config,
                device=self.generator_device,
            )
        return self._generator


class ChunkManager(BaseChunkManager):
    """BLCS-compatible wrapper around the shared chunk manager."""

    def __init__(
        self,
        *,
        chunks_dir: str | Path,
        generator_config: GeneratorConfig,
        scenes_per_chunk: int = 1000,
        epochs_per_chunk: int = 3,
        prefetch_chunks: int = 1,
        generator_device: str = "cpu",
        generation_workers: int = 0,
    ) -> None:
        self.generator_config = generator_config
        self.generator_device = generator_device
        self.generation_workers = generation_workers

        super().__init__(
            chunks_dir=chunks_dir,
            chunk_generator_factory=lambda: _BLCSChunkGenerator(
                generator_config=generator_config,
                generator_device=generator_device,
                generation_workers=generation_workers,
            ),
            scenes_per_chunk=scenes_per_chunk,
            epochs_per_chunk=epochs_per_chunk,
            prefetch_chunks=prefetch_chunks,
        )


__all__ = ["ChunkGenerator", "ChunkInfo", "ChunkManager", "ChunkState"]
