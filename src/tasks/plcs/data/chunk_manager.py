"""PLCS adapter for the shared chunk manager implementation."""

from __future__ import annotations

import threading
from pathlib import Path

from omegaconf import DictConfig

from src.tasks.base.data.chunk_manager import (
    ChunkManager as BaseChunkManager,
)
from src.tasks.plcs.court_keypoint_contract import PLCSCourtKeypointRuntimeConfig
from src.tasks.plcs.generate_dataset.config import resolve_generation_paths
from src.tasks.plcs.generate_dataset.io.dataset_io import PLCSDatasetWriter
from src.tasks.plcs.generate_dataset.utils.parallel_runner import (
    generate_parallel_scenes,
)


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
        contract = PLCSCourtKeypointRuntimeConfig.from_config(self.config).contract
        writer = PLCSDatasetWriter(
            str(chunk_dir),
            court_keypoint_contract=contract,
        )
        root_config = {"court_keypoints": {"selector": contract.selector}}
        writer.save_meta_json(config=root_config)
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
        writer.save_meta_json(config=root_config)

    def _allocate_scene_range(self, num_scenes: int) -> int:
        start_index = self._next_scene_index
        self._next_scene_index += num_scenes
        return start_index


class PLCSChunkManager(BaseChunkManager):
    """Compose the shared chunk lifecycle with PLCS scene generation."""

    def __init__(
        self,
        *,
        chunks_dir: str | Path,
        config: DictConfig,
        scenes_per_chunk: int,
        epochs_per_chunk: int,
        prefetch_chunks: int,
        generator_device: str,
        generation_workers: int,
    ) -> None:
        self.generation_workers = generation_workers
        self.config = resolve_generation_paths(config)
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


__all__ = ["PLCSChunkManager"]
