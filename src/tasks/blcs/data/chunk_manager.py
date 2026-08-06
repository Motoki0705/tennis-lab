"""BLCS adapter for the shared chunk manager implementation."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.tasks.base.data.chunk_manager import (
    ChunkManager as BaseChunkManager,
)
from src.tasks.blcs.generate_dataset.io.dataset_io import BLCSDatasetWriter
from src.tasks.blcs.generate_dataset.scene_generator import GeneratorConfig
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
        generation_chunksize: int,
        generation_seed: int,
        multi_object: bool,
        timeline_config: Mapping[str, Any] | None,
    ) -> None:
        self.generator_config = generator_config
        self.generator_device = generator_device
        self.generation_workers = generation_workers
        self.generation_chunksize = generation_chunksize
        self.generation_seed = generation_seed
        self.multi_object = multi_object
        self.timeline_config = (
            dict(timeline_config) if timeline_config is not None else None
        )
        self._next_scene_index = 0

    def __call__(
        self,
        chunk_dir: Path,
        *,
        num_scenes: int,
        stop_event: threading.Event,
    ) -> None:
        writer = BLCSDatasetWriter(str(chunk_dir))
        start_index = self._next_scene_index
        self._next_scene_index += num_scenes
        for scene_data in generate_parallel_scenes(
            generator_config=self.generator_config,
            device=self.generator_device,
            num_scenes=num_scenes,
            num_workers=self.generation_workers,
            start_index=start_index,
            seed=self.generation_seed,
            multi_object=self.multi_object,
            timeline_config=self.timeline_config,
            chunksize=self.generation_chunksize,
        ):
            if stop_event.is_set():
                break
            writer.save_scene(scene_data)


class ChunkManager(BaseChunkManager):
    """Bind BLCS scene generation to the shared chunk lifecycle."""

    def __init__(
        self,
        *,
        chunks_dir: str | Path,
        generator_config: GeneratorConfig,
        scenes_per_chunk: int,
        epochs_per_chunk: int,
        prefetch_chunks: int,
        generator_device: str,
        generation_workers: int,
        generation_chunksize: int,
        generation_seed: int,
        multi_object: bool,
        timeline_config: Mapping[str, Any] | None,
    ) -> None:
        self.generator_config = generator_config
        self.generator_device = generator_device
        self.generation_workers = generation_workers

        super().__init__(
            chunks_dir=chunks_dir,
            chunk_generator_factory=lambda: _BLCSChunkGenerator(
                generator_config=generator_config,
                generator_device=generator_device,
                generation_workers=self.generation_workers,
                generation_chunksize=generation_chunksize,
                generation_seed=generation_seed,
                multi_object=multi_object,
                timeline_config=timeline_config,
            ),
            scenes_per_chunk=scenes_per_chunk,
            epochs_per_chunk=epochs_per_chunk,
            prefetch_chunks=prefetch_chunks,
        )


__all__ = ["ChunkManager"]
