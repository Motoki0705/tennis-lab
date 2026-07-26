"""Chunk generation for canonical task scene writers."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.tasks.base.data.chunk_manager import ChunkManager


class _SceneChunkGenerator:
    def __init__(
        self,
        writer_factory: Callable[[Path], Any],
        scene_factory: Callable[[str], Any],
    ) -> None:
        self.writer_factory = writer_factory
        self.scene_factory = scene_factory
        self.next_scene_index = 0

    def __call__(
        self,
        chunk_dir: Path,
        *,
        num_scenes: int,
        stop_event: threading.Event,
    ) -> None:
        writer = self.writer_factory(chunk_dir)
        for _ in range(num_scenes):
            if stop_event.is_set():
                break
            scene_id = f"scene_train_{self.next_scene_index:09d}"
            self.next_scene_index += 1
            writer.save_scene(self.scene_factory(scene_id))


class SceneChunkManager(ChunkManager):
    """Generate fresh canonical scenes for rotating train-only chunks."""

    def __init__(
        self,
        *,
        chunks_dir: str | Path,
        writer_factory: Callable[[Path], Any],
        scene_factory: Callable[[str], Any],
        scenes_per_chunk: int,
        epochs_per_chunk: int,
        prefetch_chunks: int,
    ) -> None:
        super().__init__(
            chunks_dir=chunks_dir,
            chunk_generator_factory=lambda: _SceneChunkGenerator(
                writer_factory, scene_factory
            ),
            scenes_per_chunk=scenes_per_chunk,
            epochs_per_chunk=epochs_per_chunk,
            prefetch_chunks=prefetch_chunks,
        )


__all__ = ["SceneChunkManager"]
