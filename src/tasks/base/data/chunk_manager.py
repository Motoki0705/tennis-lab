"""Background chunk manager shared by chunked training backends.

This module owns the lifecycle of background-generated chunks. Task-specific
scene generation is injected via ``chunk_generator_factory`` so downstream
tasks can reuse the same buffering, readiness, and cleanup behavior.
"""

from __future__ import annotations

import enum
import logging
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


class ChunkGenerator(Protocol):
    """Callable that fills a chunk directory with generated scene data."""

    def __call__(
        self,
        chunk_dir: Path,
        *,
        num_scenes: int,
        stop_event: threading.Event,
    ) -> None: ...


class ChunkState(enum.Enum):
    """Lifecycle state of a generated chunk."""

    PREPARING = "preparing"
    READY = "ready"
    USED = "used"


class ChunkInfo:
    """Metadata for a single generated chunk."""

    def __init__(self, chunk_id: int, path: Path) -> None:
        self.chunk_id = chunk_id
        self.path = path
        self.state = ChunkState.PREPARING

    def __repr__(self) -> str:
        return f"ChunkInfo(id={self.chunk_id}, state={self.state.value})"


class ChunkManager:
    """Manage background generation and lifecycle of training chunks."""

    def __init__(
        self,
        *,
        chunks_dir: str | Path,
        chunk_generator_factory: Callable[[], ChunkGenerator],
        scenes_per_chunk: int = 1000,
        epochs_per_chunk: int = 3,
        prefetch_chunks: int = 1,
    ) -> None:
        self.chunks_dir = Path(chunks_dir)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir = self.chunks_dir / f"session_{uuid.uuid4().hex}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.chunk_generator_factory = chunk_generator_factory
        self.scenes_per_chunk = scenes_per_chunk
        self.epochs_per_chunk = epochs_per_chunk
        self.prefetch_chunks = prefetch_chunks

        self._chunks: dict[int, ChunkInfo] = {}
        self._next_chunk_id = 0
        self._lock = threading.Lock()
        self._ready_event = threading.Event()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

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
        """Stop the background chunk generation thread."""
        self._stop_event.set()
        self._ready_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=30)
            self._worker_thread = None
        self._cleanup_session_dir()
        logger.info("ChunkManager: background generation stopped.")

    def wait_for_ready_chunk(self, timeout: float | None = None) -> ChunkInfo | None:
        """Block until a ready chunk is available and return it."""
        while not self._stop_event.is_set():
            with self._lock:
                for info in self._chunks.values():
                    if info.state is ChunkState.READY:
                        return info
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
        threading.Thread(
            target=self._delete_chunk,
            args=(chunk_id,),
            daemon=True,
            name=f"chunk-delete-{chunk_id}",
        ).start()

    def _generation_loop(self) -> None:
        chunk_generator = self.chunk_generator_factory()

        while not self._stop_event.is_set():
            with self._lock:
                num_ready = sum(
                    1 for chunk in self._chunks.values() if chunk.state is ChunkState.READY
                )
                num_preparing = sum(
                    1
                    for chunk in self._chunks.values()
                    if chunk.state is ChunkState.PREPARING
                )
            if num_ready + num_preparing >= self.prefetch_chunks + 1:
                time.sleep(1.0)
                continue

            chunk_id = self._allocate_chunk_id()
            chunk_dir = self.session_dir / f"scene_{chunk_id}"
            chunk_dir.mkdir(parents=True, exist_ok=True)

            info = ChunkInfo(chunk_id=chunk_id, path=chunk_dir)
            with self._lock:
                self._chunks[chunk_id] = info

            logger.info(
                "ChunkManager: generating chunk %d (%d scenes) -> %s",
                chunk_id,
                self.scenes_per_chunk,
                chunk_dir,
            )

            try:
                chunk_generator(
                    info.path,
                    num_scenes=self.scenes_per_chunk,
                    stop_event=self._stop_event,
                )
                if self._stop_event.is_set():
                    with self._lock:
                        info.state = ChunkState.USED
                    self._delete_chunk(chunk_id)
                    break
                self._write_train_split(info.path)
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

    def _write_train_split(self, chunk_dir: Path) -> None:
        scenes_dir = chunk_dir / "scenes"
        scene_files = sorted(path.name for path in scenes_dir.glob("*") if path.is_dir())
        if not scene_files:
            raise RuntimeError(f"No scenes were generated in {scenes_dir}")
        train_file = chunk_dir / "train.txt"
        train_file.write_text("\n".join(scene_files) + "\n", encoding="utf-8")

    def _allocate_chunk_id(self) -> int:
        with self._lock:
            chunk_id = self._next_chunk_id
            self._next_chunk_id += 1
        return chunk_id

    def _delete_chunk(self, chunk_id: int) -> None:
        with self._lock:
            info = self._chunks.pop(chunk_id, None)
        if info is None:
            return
        try:
            if info.path.exists():
                shutil.rmtree(info.path)
                logger.info("ChunkManager: deleted chunk %d at %s", chunk_id, info.path)
        except FileNotFoundError:
            return
        except OSError:
            logger.exception("ChunkManager: error deleting chunk %d", chunk_id)

    def _cleanup_session_dir(self) -> None:
        with self._lock:
            self._chunks.clear()
        try:
            if self.session_dir.exists():
                shutil.rmtree(self.session_dir)
        except OSError:
            logger.exception(
                "ChunkManager: error deleting chunk session dir %s",
                self.session_dir,
            )
