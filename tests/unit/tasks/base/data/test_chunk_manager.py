"""Unit tests for the background chunk manager.

These exercise the deterministic, single-threaded pieces (id allocation,
split-file writing, used-marking, session-dir cleanup) and one short
end-to-end background-generation cycle with a fast synchronous generator. The
generator writes scene dirs immediately so ``wait_for_ready_chunk`` returns
quickly; timeouts keep the suite from hanging if something regresses.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from src.tasks.base.data.chunk_manager import (
    ChunkInfo,
    ChunkManager,
    ChunkState,
)

pytestmark = pytest.mark.unit


def _fast_generator_factory(scene_count: int = 2):
    """Generator that writes ``scene_count`` empty scene dirs synchronously."""

    def _generate(chunk_dir: Path, *, num_scenes: int, stop_event: threading.Event) -> None:
        scenes = chunk_dir / "scenes"
        scenes.mkdir(parents=True, exist_ok=True)
        for i in range(scene_count):
            (scenes / f"scene_{i:03d}").mkdir()

    return lambda: _generate


def _make_manager(tmp_path: Path, **kw) -> ChunkManager:
    return ChunkManager(
        chunks_dir=tmp_path / "chunks",
        chunk_generator_factory=_fast_generator_factory(kw.pop("scene_count", 2)),
        scenes_per_chunk=kw.pop("scenes_per_chunk", 2),
        epochs_per_chunk=kw.pop("epochs_per_chunk", 1),
        prefetch_chunks=kw.pop("prefetch_chunks", 0),
    )


def test_init_creates_session_dir(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    assert mgr.session_dir.exists()
    assert mgr.session_dir.parent == mgr.chunks_dir
    assert mgr.session_dir.name.startswith("session_")


def test_allocate_chunk_id_is_monotonic(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    ids = [mgr._allocate_chunk_id() for _ in range(4)]
    assert ids == [0, 1, 2, 3]
    assert mgr._next_chunk_id == 4


def test_write_train_split_lists_scene_dirs(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    chunk_dir = tmp_path / "c0"
    scenes = chunk_dir / "scenes"
    scenes.mkdir(parents=True)
    (scenes / "scene_001").mkdir()
    (scenes / "scene_000").mkdir()
    (scenes / "not_a_dir.txt").write_text("x")  # files are ignored

    mgr._write_train_split(chunk_dir)
    lines = (chunk_dir / "train.txt").read_text().splitlines()
    assert lines == ["scene_000", "scene_001"]  # sorted, dirs only


def test_write_train_split_raises_when_empty(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    chunk_dir = tmp_path / "empty"
    (chunk_dir / "scenes").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="No scenes were generated"):
        mgr._write_train_split(chunk_dir)


def test_mark_used_on_unknown_id_is_noop(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    # Should not raise even though id 99 was never registered.
    mgr.mark_used(99)


def test_delete_chunk_removes_dir_and_entry(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    chunk_dir = mgr.session_dir / "scene_0"
    chunk_dir.mkdir(parents=True)
    mgr._chunks[0] = ChunkInfo(chunk_id=0, path=chunk_dir)

    mgr._delete_chunk(0)
    assert 0 not in mgr._chunks
    assert not chunk_dir.exists()


def test_wait_for_ready_chunk_times_out_when_none(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    # No worker started, so no ready chunk -> returns None within the timeout.
    assert mgr.wait_for_ready_chunk(timeout=0.2) is None


def test_chunk_info_repr() -> None:
    info = ChunkInfo(chunk_id=3, path=Path("/x"))
    assert "id=3" in repr(info)
    assert info.state is ChunkState.PREPARING


def test_background_generation_cycle(tmp_path: Path) -> None:
    """One start->ready->mark_used->stop cycle with a fast generator."""
    mgr = _make_manager(tmp_path, prefetch_chunks=0, scene_count=2)
    mgr.start()
    try:
        chunk = mgr.wait_for_ready_chunk(timeout=10.0)
        assert chunk is not None
        assert chunk.state is ChunkState.READY
        assert (chunk.path / "train.txt").exists()
        assert (chunk.path / "scenes").is_dir()
        mgr.mark_used(chunk.chunk_id)
    finally:
        mgr.stop()
    # After stop, the session dir is cleaned up.
    assert not mgr.session_dir.exists()


def test_stop_cleans_session_dir(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    assert mgr.session_dir.exists()
    mgr.stop()
    assert not mgr.session_dir.exists()


def test_generation_stop_does_not_validate_an_intentionally_partial_chunk(
    tmp_path: Path,
) -> None:
    def _stop_during_generation(
        chunk_dir: Path,
        *,
        num_scenes: int,
        stop_event: threading.Event,
    ) -> None:
        del chunk_dir, num_scenes
        stop_event.set()

    mgr = ChunkManager(
        chunks_dir=tmp_path / "chunks",
        chunk_generator_factory=lambda: _stop_during_generation,
        scenes_per_chunk=2,
        prefetch_chunks=0,
    )

    mgr._generation_loop()

    assert not mgr._chunks
