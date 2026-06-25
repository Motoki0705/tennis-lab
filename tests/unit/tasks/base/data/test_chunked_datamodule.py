"""Unit tests for BaseChunkedDataModule chunk-rotation lifecycle."""

from __future__ import annotations

from pathlib import Path

import pytest
from torch.utils.data import Dataset

from src.tasks.base.data.chunked_datamodule import BaseChunkedDataModule

pytestmark = pytest.mark.unit


class _FakeChunk:
    def __init__(self, chunk_id: int) -> None:
        self.chunk_id = chunk_id
        self.path = Path(f"/chunks/scene_{chunk_id}")


class _FakeChunkManager:
    """Records start/stop/mark_used and serves chunks in sequence."""

    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.used: list[int] = []
        self._next = 0

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def mark_used(self, chunk_id: int) -> None:
        self.used.append(chunk_id)

    def wait_for_ready_chunk(self):
        chunk = _FakeChunk(self._next)
        self._next += 1
        return chunk


class _DummyDataset(Dataset):
    def __init__(self, scene_dir) -> None:
        self.scene_dir = scene_dir

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx):
        return idx


class _DM(BaseChunkedDataModule):
    def _build_collate_fn(self):
        return None

    def _build_dataset(self, scene_dir, split_file, augment):
        return _DummyDataset(scene_dir)

    def _dataset_name(self) -> str:
        return "dummy"

    def _build_chunk_manager(self):
        return _FakeChunkManager()

    def _default_chunks_dir(self) -> str:
        return "outputs/chunks"


def test_chunk_config_parsing() -> None:
    dm = _DM(
        {
            "data": {
                "chunk": {
                    "scenes_per_chunk": 500,
                    "epochs_per_chunk": 2,
                    "prefetch_chunks": 3,
                    "chunks_dir": "/tmp/chunks",
                    "generation_workers": 4,
                },
                "generator_device": "cuda",
            }
        }
    )
    assert dm.scenes_per_chunk == 500
    assert dm.epochs_per_chunk == 2
    assert dm.prefetch_chunks == 3
    assert dm.chunks_dir == Path("/tmp/chunks")
    assert dm.generation_workers == 4
    assert dm.generator_device == "cuda"


def test_chunk_config_defaults() -> None:
    dm = _DM({})
    assert dm.scenes_per_chunk == 1000
    assert dm.epochs_per_chunk == 3
    assert dm.prefetch_chunks == 1
    assert dm.chunks_dir == Path("outputs/chunks")
    assert dm.generator_device == "cpu"


def test_epoch_end_rotates_after_epochs_per_chunk(tmp_path: Path) -> None:
    root = tmp_path / "scenes"
    root.mkdir()
    for s in ("train", "val", "test"):
        (root / f"{s}.txt").write_text("scene_0000\n", encoding="utf-8")

    dm = _DM({"data": {"scene_dir": str(root), "chunk": {"epochs_per_chunk": 2}}})
    dm.setup("fit")
    assert dm.chunk_manager.started is True  # type: ignore[union-attr]

    # epoch 1: counter 1 < 2 -> no rotation
    dm.on_train_epoch_end()
    assert dm._current_chunk_id is None
    assert dm._epochs_on_current_chunk == 1

    # epoch 2: counter 2 >= 2 -> rotate (first chunk loaded, none marked used)
    dm.on_train_epoch_end()
    assert dm._current_chunk_id == 0
    assert dm._epochs_on_current_chunk == 0
    assert dm.chunk_manager.used == []  # type: ignore[union-attr]

    # next two epochs -> rotate again, marking the old chunk used
    dm.on_train_epoch_end()
    dm.on_train_epoch_end()
    assert dm._current_chunk_id == 1
    assert dm.chunk_manager.used == [0]  # type: ignore[union-attr]


def test_teardown_stops_manager(tmp_path: Path) -> None:
    root = tmp_path / "scenes"
    root.mkdir()
    for s in ("train", "val", "test"):
        (root / f"{s}.txt").write_text("scene_0000\n", encoding="utf-8")
    dm = _DM({"data": {"scene_dir": str(root)}})
    dm.setup("fit")
    manager = dm.chunk_manager
    dm.teardown("fit")
    assert manager.stopped is True  # type: ignore[union-attr]
    assert dm.chunk_manager is None


def test_load_next_chunk_raises_when_none(tmp_path: Path) -> None:
    class _NoneManager(_FakeChunkManager):
        def wait_for_ready_chunk(self):
            return None

    class _DMNone(_DM):
        def _build_chunk_manager(self):
            return _NoneManager()

    root = tmp_path / "scenes"
    root.mkdir()
    for s in ("train", "val", "test"):
        (root / f"{s}.txt").write_text("scene_0000\n", encoding="utf-8")
    dm = _DMNone({"data": {"scene_dir": str(root)}})
    dm.setup("fit")
    with pytest.raises(RuntimeError, match="no ready chunk"):
        dm._load_next_chunk()
