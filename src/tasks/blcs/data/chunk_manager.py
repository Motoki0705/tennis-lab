"""Background chunk manager for on-the-fly BLCS training.

Generates scene chunks in a background thread and manages their lifecycle.
Each chunk is a directory containing NPZ scene files that can be loaded by
``BallTrajectoryDataset``.  Chunks cycle through three states:

- **preparing** – being generated in the background.
- **ready** – generation complete, available for training.
- **used** – training has consumed it; will be deleted.
"""

from __future__ import annotations

import enum
import logging
import random
import shutil
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from src.tasks.blcs.generate_dataset.scene_generator import (
        BLCSSceneGenerator,
        GeneratorConfig,
    )

logger = logging.getLogger(__name__)


class ChunkState(enum.Enum):
    """Lifecycle state of a scene chunk."""

    PREPARING = "preparing"
    READY = "ready"
    USED = "used"


class ChunkInfo:
    """Metadata for a single chunk."""

    def __init__(self, chunk_id: int, path: Path) -> None:
        self.chunk_id = chunk_id
        self.path = path
        self.state = ChunkState.PREPARING

    def __repr__(self) -> str:
        return f"ChunkInfo(id={self.chunk_id}, state={self.state.value})"


class ChunkManager:
    """Manages background generation and lifecycle of scene chunks.

    Parameters
    ----------
    chunks_dir:
        Root directory for chunks (e.g. ``data/blcs/chunks``).
    generator_config:
        Configuration forwarded to :class:`BLCSSceneGenerator`.
    scenes_per_chunk:
        Number of scenes to generate per chunk.
    epochs_per_chunk:
        How many epochs to reuse a chunk before switching.
    prefetch_chunks:
        How many chunks to keep ready ahead of training.
    generator_device:
        Device for the scene generator (``"cpu"`` or ``"cuda"``).
    generation_workers:
        Number of parallel worker processes for scene generation.
        ``0`` (default) uses sequential generation in the background thread.
        This is independent of the DataLoader ``num_workers``.
    """

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
        self.chunks_dir = Path(chunks_dir)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.generator_config = generator_config
        self.scenes_per_chunk = scenes_per_chunk
        self.epochs_per_chunk = epochs_per_chunk
        self.prefetch_chunks = prefetch_chunks
        self.generator_device = generator_device
        self.generation_workers = generation_workers

        self._chunks: dict[int, ChunkInfo] = {}
        self._next_chunk_id = 0
        self._lock = threading.Lock()
        self._ready_event = threading.Event()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background chunk generation thread."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._generation_loop,
            daemon=True,
            name="chunk-generator",
        )
        self._worker_thread.start()
        logger.info("ChunkManager: background generation started.")

    def stop(self) -> None:
        """Stop the background generation thread."""
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=30)
            self._worker_thread = None
        logger.info("ChunkManager: background generation stopped.")

    def wait_for_ready_chunk(self, timeout: float | None = None) -> ChunkInfo | None:
        """Block until a ready chunk is available and return it.

        The returned chunk's state is immediately set to ``READY`` (it stays
        ready while training consumes it; call :meth:`mark_used` when done).
        """
        while not self._stop_event.is_set():
            with self._lock:
                for info in self._chunks.values():
                    if info.state is ChunkState.READY:
                        return info
            # Wait for the background thread to signal a new ready chunk.
            signalled = self._ready_event.wait(timeout=timeout or 5.0)
            self._ready_event.clear()
            if timeout is not None and not signalled:
                return None
        return None

    def mark_used(self, chunk_id: int) -> None:
        """Mark a chunk as used and schedule its deletion."""
        with self._lock:
            info = self._chunks.get(chunk_id)
            if info is None:
                return
            info.state = ChunkState.USED
        # Delete the chunk directory in a background thread to avoid blocking.
        threading.Thread(
            target=self._delete_chunk,
            args=(chunk_id,),
            daemon=True,
            name=f"chunk-delete-{chunk_id}",
        ).start()

    # ------------------------------------------------------------------
    # Background generation
    # ------------------------------------------------------------------

    def _generation_loop(self) -> None:
        """Continuously generate chunks while the manager is running."""
        from src.tasks.blcs.generate_dataset.io.dataset_io import BLCSDatasetWriter
        from src.tasks.blcs.generate_dataset.scene_generator import BLCSSceneGenerator

        generator = BLCSSceneGenerator(
            config=self.generator_config,
            device=self.generator_device,
        )

        while not self._stop_event.is_set():
            # Check if we need more chunks
            with self._lock:
                num_ready = sum(
                    1 for c in self._chunks.values() if c.state is ChunkState.READY
                )
                num_preparing = sum(
                    1 for c in self._chunks.values() if c.state is ChunkState.PREPARING
                )
            if num_ready + num_preparing >= self.prefetch_chunks + 1:
                # Enough chunks buffered; sleep briefly and re-check.
                time.sleep(1.0)
                continue

            chunk_id = self._allocate_chunk_id()
            chunk_dir = self.chunks_dir / f"scene_{chunk_id}"
            chunk_dir.mkdir(parents=True, exist_ok=True)

            info = ChunkInfo(chunk_id=chunk_id, path=chunk_dir)
            with self._lock:
                self._chunks[chunk_id] = info

            logger.info(
                "ChunkManager: generating chunk %d (%d scenes) → %s",
                chunk_id,
                self.scenes_per_chunk,
                chunk_dir,
            )

            try:
                self._generate_chunk(generator, info)
            except Exception:
                logger.exception("ChunkManager: failed to generate chunk %d", chunk_id)
                with self._lock:
                    info.state = ChunkState.USED
                self._delete_chunk(chunk_id)
                continue

            with self._lock:
                info.state = ChunkState.READY
            self._ready_event.set()
            logger.info("ChunkManager: chunk %d ready.", chunk_id)

    def _generate_chunk(
        self,
        generator: BLCSSceneGenerator,
        info: ChunkInfo,
    ) -> None:
        """Generate scenes for a single chunk and write NPZ files."""
        from src.tasks.blcs.generate_dataset.io.dataset_io import BLCSDatasetWriter

        writer = BLCSDatasetWriter(str(info.path))
        scene_count = 0

        if self.generation_workers > 0:
            from src.tasks.blcs.generate_dataset.utils.parallel_runner import (
                generate_parallel_scenes,
            )

            for scene_data in generate_parallel_scenes(
                generator_config=self.generator_config,
                device=self.generator_device,
                num_scenes=self.scenes_per_chunk,
                num_workers=self.generation_workers,
            ):
                if self._stop_event.is_set():
                    break
                writer.save_scene(scene_data)
                scene_count += 1
        else:
            for scene_data in generator.generate(self.scenes_per_chunk):
                if self._stop_event.is_set():
                    break
                writer.save_scene(scene_data)
                scene_count += 1

        # Write a train.txt split file listing all scenes (used by the dataset).
        scene_files = sorted(p.name for p in info.path.glob("scenes/scene_*") if p.is_dir())
        train_file = info.path / "train.txt"
        train_file.write_text("\n".join(scene_files) + "\n")

        logger.info(
            "ChunkManager: chunk %d finished – %d scenes written.",
            info.chunk_id,
            scene_count,
        )

    def _allocate_chunk_id(self) -> int:
        with self._lock:
            cid = self._next_chunk_id
            self._next_chunk_id += 1
        return cid

    def _delete_chunk(self, chunk_id: int) -> None:
        with self._lock:
            info = self._chunks.pop(chunk_id, None)
        if info is None:
            return
        try:
            if info.path.exists():
                shutil.rmtree(info.path)
                logger.info("ChunkManager: deleted chunk %d at %s", chunk_id, info.path)
        except OSError:
            logger.exception("ChunkManager: error deleting chunk %d", chunk_id)
