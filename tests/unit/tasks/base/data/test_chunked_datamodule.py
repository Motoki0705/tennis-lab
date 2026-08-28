"""Unit tests for BaseChunkedDataModule chunk-rotation lifecycle."""

from __future__ import annotations

from pathlib import Path

import pytest
from torch.utils.data import Dataset

from src.tasks.base.data.chunked_datamodule import BaseChunkedDataModule
from src.utils.configuration import MissingConfigurationKeyError

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

    def seed_worker(self, *, worker_seed: int, worker_id: int) -> None:
        del worker_seed, worker_id


class _DM(BaseChunkedDataModule):
    def _build_collate_fn(self):
        return None

    def _build_dataset(self, scene_dir, split_file, augment, seed=None):
        del split_file, augment, seed
        return _DummyDataset(scene_dir)

    def _dataset_name(self) -> str:
        return "dummy"

    def _build_chunk_manager(self):
        return _FakeChunkManager()


def _config(
    root: Path,
    *,
    scenes_per_chunk: int = 1000,
    epochs_per_chunk: int = 3,
    prefetch_chunks: int = 1,
    generation_workers: int = 1,
    generator_device: str = "cpu",
) -> dict[str, object]:
    return {
        "paths": {
            "project_root": str(root.parent),
            "data_root": str(root.parent),
            "checkpoint_root": "checkpoints",
            "artifact_root": str(root.parent),
            "output_root": "outputs",
            "cache_root": ".cache",
            "external_asset_root": "external",
        },
        "data": {
            "scene_dir": root.name,
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "chunk": {
                "scenes_per_chunk": scenes_per_chunk,
                "epochs_per_chunk": epochs_per_chunk,
                "prefetch_chunks": prefetch_chunks,
                "chunks_dir": "chunks",
                "generation_workers": generation_workers,
            },
            "generator_device": generator_device,
        },
        "run": {"seed": 42},
    }


def test_chunk_config_parsing(tmp_path: Path) -> None:
    scene_root = tmp_path / "dataset_fixture"
    dm = _DM(
        _config(
            scene_root,
            scenes_per_chunk=500,
            epochs_per_chunk=2,
            prefetch_chunks=3,
            generation_workers=4,
            generator_device="cuda",
        )
    )
    assert dm.scenes_per_chunk == 500
    assert dm.epochs_per_chunk == 2
    assert dm.prefetch_chunks == 3
    assert dm.chunks_dir == tmp_path / "chunks"
    assert dm.generation_workers == 4
    assert dm.generator_device == "cuda"


def test_chunk_config_allows_zero_prefetch(tmp_path: Path) -> None:
    dm = _DM(_config(tmp_path / "dataset_fixture", prefetch_chunks=0))

    assert dm.prefetch_chunks == 0


def test_chunk_config_rejects_missing_contract() -> None:
    with pytest.raises(MissingConfigurationKeyError, match="configuration.paths"):
        _DM({})


def test_epoch_end_rotates_after_epochs_per_chunk(tmp_path: Path) -> None:
    root = tmp_path / "scenes"
    root.mkdir()
    for s in ("train", "val", "test"):
        (root / f"{s}.txt").write_text("scene_0000\n", encoding="utf-8")

    dm = _DM(_config(root, epochs_per_chunk=2))
    dm.setup("fit")
    assert dm.chunk_manager.started is True  # type: ignore[union-attr]
    assert dm._current_chunk_id == 0
    assert dm._epochs_on_current_chunk == 0

    # epoch 1: counter 1 < 2 -> keep the initial generated chunk
    dm.on_train_epoch_end()
    assert dm._current_chunk_id == 0
    assert dm._epochs_on_current_chunk == 1

    # epoch 2: counter 2 >= 2 -> rotate and retire the initial chunk
    dm.on_train_epoch_end()
    assert dm._current_chunk_id == 1
    assert dm._epochs_on_current_chunk == 0
    assert dm.chunk_manager.used == [0]  # type: ignore[union-attr]

    # next two epochs -> rotate again, marking the old chunk used
    dm.on_train_epoch_end()
    dm.on_train_epoch_end()
    assert dm._current_chunk_id == 2
    assert dm.chunk_manager.used == [0, 1]  # type: ignore[union-attr]


def test_teardown_stops_manager(tmp_path: Path) -> None:
    root = tmp_path / "scenes"
    root.mkdir()
    for s in ("train", "val", "test"):
        (root / f"{s}.txt").write_text("scene_0000\n", encoding="utf-8")
    dm = _DM(_config(root))
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
    dm = _DMNone(_config(root))
    with pytest.raises(RuntimeError, match="no ready chunk"):
        dm.setup("fit")
