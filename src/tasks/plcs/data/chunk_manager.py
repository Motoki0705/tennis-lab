"""PLCS adapter for the shared chunk manager implementation."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from src.tasks.base.data.chunk_manager import (
    ChunkGenerator,
    ChunkInfo,
    ChunkManager as BaseChunkManager,
    ChunkState,
)
from src.tasks.plcs.generate_dataset.io.dataset_io import PLCSDatasetWriter
from src.tasks.plcs.generate_dataset.sampling.motion_sampler import MotionSampler
from src.tasks.plcs.generate_dataset.scene_generator import SceneGenerator

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def _resolve_generator_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class _PLCSChunkGenerator:
    def __init__(
        self,
        *,
        config: DictConfig,
        generator_device: str,
        category: str | None,
        max_attempt_factor: int,
    ) -> None:
        self.config = config
        self.category = category
        self.max_attempt_factor = max_attempt_factor
        self.resolved_device = _resolve_generator_device(generator_device)
        self._motion_sampler: MotionSampler | None = None
        self._scene_generator: SceneGenerator | None = None
        self._next_scene_index = 0

    def __call__(
        self,
        chunk_dir: Path,
        *,
        num_scenes: int,
        stop_event: threading.Event,
    ) -> None:
        writer = PLCSDatasetWriter(str(chunk_dir))
        successful = 0
        attempts = 0
        max_attempts = max(num_scenes * self.max_attempt_factor, num_scenes)
        scene_generator = self._get_scene_generator()

        while successful < num_scenes:
            if stop_event.is_set():
                break
            if attempts >= max_attempts:
                raise RuntimeError(
                    "PLCS chunk generation could not produce enough valid scenes. "
                    f"Generated {successful}/{num_scenes} after {attempts} attempts."
                )

            scene_id = self._allocate_scene_id()
            attempts += 1

            try:
                scene = scene_generator.generate_scene(
                    scene_id=scene_id,
                    category=self.category,
                )
            except Exception:
                logger.exception("PLCS chunk generation failed for %s", scene_id)
                continue

            if not scene.cameras:
                logger.warning(
                    "Skipping PLCS chunk scene %s because it has no valid cameras.",
                    scene_id,
                )
                continue

            writer.save_scene(scene)
            successful += 1

    def _get_motion_sampler(self) -> MotionSampler:
        if self._motion_sampler is None:
            self._motion_sampler = MotionSampler(
                config=self.config,
                smplh_model_path=str(self.config.paths.smplh_model_path),
                device=self.resolved_device,
            )
        return self._motion_sampler

    def _get_scene_generator(self) -> SceneGenerator:
        if self._scene_generator is None:
            self._scene_generator = SceneGenerator(
                config=self.config,
                motion_sampler=self._get_motion_sampler(),
                device=self.resolved_device,
            )
        return self._scene_generator

    def _allocate_scene_id(self) -> str:
        scene_id = f"scene_{self._next_scene_index:06d}"
        self._next_scene_index += 1
        return scene_id


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
        generator_device: str = "auto",
        generation_workers: int = 0,
        category: str | None = None,
        max_attempt_factor: int = 10,
    ) -> None:
        if generation_workers > 0:
            raise ValueError("PLCS chunk generation does not support generation_workers > 0.")

        self.config = config
        self.generator_device = generator_device
        self.category = category
        self.max_attempt_factor = max_attempt_factor

        super().__init__(
            chunks_dir=chunks_dir,
            chunk_generator_factory=lambda: _PLCSChunkGenerator(
                config=config,
                generator_device=generator_device,
                category=category,
                max_attempt_factor=max_attempt_factor,
            ),
            scenes_per_chunk=scenes_per_chunk,
            epochs_per_chunk=epochs_per_chunk,
            prefetch_chunks=prefetch_chunks,
        )


__all__ = ["ChunkGenerator", "ChunkInfo", "ChunkManager", "ChunkState"]