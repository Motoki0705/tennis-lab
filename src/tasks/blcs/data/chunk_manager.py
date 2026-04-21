"""BLCS adapter for the shared chunk manager implementation."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

from src.tasks.base.data.chunk_manager import (
    ChunkGenerator,
    ChunkInfo,
    ChunkState,
)
from src.tasks.base.data.chunk_manager import (
    ChunkManager as BaseChunkManager,
)

if TYPE_CHECKING:
    from src.tasks.blcs.generate_dataset.scene_generator import (
        BLCSSceneGenerator,
        GeneratorConfig,
    )


def _build_chunk_generator(
    *,
    generator_config: GeneratorConfig,
    generator_device: str,
    generation_workers: int,
) -> ChunkGenerator:
    generator: BLCSSceneGenerator | None = None

    def generate_chunk(
        chunk_dir: Path,
        *,
        num_scenes: int,
        stop_event: threading.Event,
    ) -> None:
        nonlocal generator

        from src.tasks.blcs.generate_dataset.io.dataset_io import BLCSDatasetWriter

        writer = BLCSDatasetWriter(str(chunk_dir))
        if generation_workers > 0:
            from src.tasks.blcs.generate_dataset.utils.parallel_runner import (
                generate_parallel_scenes,
            )

            for scene_data in generate_parallel_scenes(
                generator_config=generator_config,
                device=generator_device,
                num_scenes=num_scenes,
                num_workers=generation_workers,
            ):
                if stop_event.is_set():
                    break
                writer.save_scene(scene_data)
            return

        if generator is None:
            from src.tasks.blcs.generate_dataset.scene_generator import (
                BLCSSceneGenerator,
            )

            generator = BLCSSceneGenerator(
                config=generator_config,
                device=generator_device,
            )

        for scene_data in generator.generate(num_scenes):
            if stop_event.is_set():
                break
            writer.save_scene(scene_data)

    return generate_chunk


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
            chunk_generator_factory=lambda: _build_chunk_generator(
                generator_config=generator_config,
                generator_device=generator_device,
                generation_workers=generation_workers,
            ),
            scenes_per_chunk=scenes_per_chunk,
            epochs_per_chunk=epochs_per_chunk,
            prefetch_chunks=prefetch_chunks,
        )


__all__ = ["ChunkGenerator", "ChunkInfo", "ChunkManager", "ChunkState"]
