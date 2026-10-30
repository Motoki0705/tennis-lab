"""Unit tests for the base dataset writer (split + meta generation)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from src.tasks.base.data.dataset_writer import BaseDatasetWriter

pytestmark = pytest.mark.unit


class _Meta:
    def __init__(self, d: dict) -> None:
        self._d = d

    def to_dict(self) -> dict:
        return self._d


class _Writer(BaseDatasetWriter):
    def save_scene(self, scene_data) -> Path:  # pragma: no cover - not exercised here
        raise NotImplementedError


def _writer_with_records(tmp_path: Path, n: int) -> _Writer:
    w = _Writer(tmp_path / "out")
    for i in range(n):
        w.scene_records.append({"file": f"scene_{i:04d}", "num_cameras": (i % 3) + 1})
    return w


def test_init_creates_dirs(tmp_path: Path) -> None:
    w = _Writer(tmp_path / "out")
    assert w.output_dir.exists()
    assert w.scenes_dir.exists()
    assert w.scenes_dir.name == "scenes"


@pytest.mark.parametrize(
    ("n_scenes", "ratios", "expected_counts"),
    [
        (10, (0.6, 0.2, 0.2), (6, 2, 2)),
        # Equal fractional remainders are resolved in train/val/test order.
        (5, (0.8, 0.1, 0.1), (4, 1, 0)),
        # test gets two scenes; the largest remaining fraction belongs to val.
        (3, (0.1, 0.2, 0.7), (0, 1, 2)),
        (2, (0.5, 0.5, 0.0), (1, 1, 0)),
    ],
)
def test_save_split_info_partition_sizes(
    tmp_path: Path,
    n_scenes: int,
    ratios: tuple[float, float, float],
    expected_counts: tuple[int, int, int],
) -> None:
    w = _writer_with_records(tmp_path, n_scenes)
    w.save_split_info(
        train_ratio=ratios[0], val_ratio=ratios[1], test_ratio=ratios[2], seed=0
    )

    train = (w.output_dir / "train.txt").read_text().splitlines()
    val = (w.output_dir / "val.txt").read_text().splitlines()
    test = (w.output_dir / "test.txt").read_text().splitlines()
    assert (len(train), len(val), len(test)) == expected_counts
    # Every scene is accounted for exactly once, including across train/test.
    assert set(train) | set(val) | set(test) == {
        f"scene_{i:04d}" for i in range(n_scenes)
    }
    assert not (set(train) & set(val))
    assert not (set(val) & set(test))
    assert not (set(train) & set(test))
    info = json.loads((w.output_dir / "split_info.json").read_text())
    assert info["n_scenes"] == dict(
        zip(("train", "val", "test"), expected_counts, strict=True)
    )


@pytest.mark.parametrize(
    ("ratios", "message"),
    [
        ((float("nan"), 0.1, 0.1), "train_ratio.*finite"),
        ((0.8, float("inf"), 0.1), "val_ratio.*finite"),
        ((0.8, 0.1, float("-inf")), "test_ratio.*finite"),
        ((0.8, 0.8, -0.6), "test_ratio.*non-negative"),
        ((0.6, 0.2, 0.1), "sum to 1"),
    ],
)
def test_save_split_info_rejects_invalid_ratios_before_writing(
    tmp_path: Path, ratios: tuple[float, float, float], message: str
) -> None:
    w = _writer_with_records(tmp_path, 10)
    paths = [
        w.output_dir / name
        for name in ("train.txt", "val.txt", "test.txt", "split_info.json")
    ]
    for path in paths:
        path.write_text("previous split data\n")

    with pytest.raises(ValueError, match=message):
        w.save_split_info(
            train_ratio=ratios[0], val_ratio=ratios[1], test_ratio=ratios[2]
        )

    assert all(path.read_text() == "previous split data\n" for path in paths)


def test_save_split_info_rejects_duplicate_scene_ids_before_writing(
    tmp_path: Path,
) -> None:
    w = _writer_with_records(tmp_path, 3)
    for record in w.scene_records:
        record["file"] = "same_scene"

    with pytest.raises(ValueError, match="Duplicate scene ID.*same_scene"):
        w.save_split_info(train_ratio=1 / 3, val_ratio=1 / 3, test_ratio=1 / 3)

    assert list(w.output_dir.iterdir()) == [w.scenes_dir]


def test_save_split_info_is_deterministic(tmp_path: Path) -> None:
    w1 = _writer_with_records(tmp_path / "a", 8)
    w2 = _writer_with_records(tmp_path / "b", 8)
    state = random.getstate()
    records = [record.copy() for record in w1.scene_records]
    try:
        w1.save_split_info(seed=123)
        assert random.getstate() == state
        random.random()  # Reproducibility must not depend on global random state.
        w2.save_split_info(seed=123)
        for name in ("train.txt", "val.txt", "test.txt"):
            assert (w1.output_dir / name).read_text() == (
                w2.output_dir / name
            ).read_text()
        assert w1.scene_records == records
    finally:
        random.setstate(state)


def test_save_split_info_writes_split_info_json(tmp_path: Path) -> None:
    w = _writer_with_records(tmp_path, 5)
    w.save_split_info(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=7)
    info = json.loads((w.output_dir / "split_info.json").read_text())
    assert info["seed"] == 7
    assert info["train_ratio"] == 0.8
    assert sum(info["n_scenes"].values()) == 5


def test_save_meta_json_stats(tmp_path: Path) -> None:
    w = _writer_with_records(tmp_path, 3)  # num_cameras: 1, 2, 3
    w.save_meta_json(config={"k": "v"})
    meta = json.loads((w.output_dir / "meta.json").read_text())
    assert meta["stats"]["total_scenes"] == 3
    assert meta["stats"]["total_cameras"] == 6
    assert meta["stats"]["avg_cameras_per_scene"] == pytest.approx(2.0)
    assert meta["config"] == {"k": "v"}
    assert len(meta["scenes"]) == 3


def test_save_meta_json_empty_records(tmp_path: Path) -> None:
    w = _Writer(tmp_path / "out")
    w.save_meta_json()
    meta = json.loads((w.output_dir / "meta.json").read_text())
    assert meta["stats"]["total_scenes"] == 0
    assert meta["stats"]["avg_cameras_per_scene"] == 0


def test_write_scene_files_roundtrip(tmp_path: Path) -> None:
    w = _Writer(tmp_path / "out")
    scene_path = w.scenes_dir / "scene_0000"
    scene_path.mkdir()
    arrays: dict[str, np.ndarray] = {
        "position": np.arange(6).reshape(3, 2).astype(np.float32)
    }
    w._write_scene_files(
        scene_path,
        scene_meta=_Meta({"num_frames": 3}),
        scalars={"num_cameras": 2},
        arrays=arrays,
    )
    assert json.loads((scene_path / "meta.json").read_text()) == {"num_frames": 3}
    assert json.loads((scene_path / "scalars.json").read_text()) == {"num_cameras": 2}
    loaded = np.load(scene_path / "position.npy")
    assert np.array_equal(loaded, arrays["position"])
