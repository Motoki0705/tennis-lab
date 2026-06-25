"""Unit tests for SceneDirectoryDataModule setup/validation logic."""

from __future__ import annotations

from pathlib import Path

import pytest
from torch.utils.data import Dataset

from src.tasks.base.data.datamodule import SceneDirectoryDataModule

pytestmark = pytest.mark.unit


class _DummyDataset(Dataset):
    def __init__(self, name: str, n: int = 4) -> None:
        self.name = name
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return idx


class _DM(SceneDirectoryDataModule):
    def _build_collate_fn(self):
        return None

    def _build_dataset(self, scene_dir, split_file, augment):
        return _DummyDataset(f"{split_file}:{augment}")

    def _dataset_name(self) -> str:
        return "dummy"


def _make_scene_root(tmp_path: Path, *, splits: tuple[str, ...]) -> Path:
    root = tmp_path / "scene_dir"
    root.mkdir()
    for s in splits:
        (root / f"{s}.txt").write_text("scene_0000\n", encoding="utf-8")
    return root


def test_config_parsing_defaults() -> None:
    dm = _DM({})
    assert dm.batch_size == _DM.default_batch_size
    assert dm.num_workers == 4
    assert dm.pin_memory is True
    assert dm.scene_dir == Path(_DM.default_scene_dir)


def test_config_parsing_overrides() -> None:
    dm = _DM({"data": {"batch_size": 8, "num_workers": 2, "pin_memory": False, "scene_dir": "/x"}})
    assert dm.batch_size == 8
    assert dm.num_workers == 2
    assert dm.pin_memory is False
    assert dm.scene_dir == Path("/x")


def test_setup_missing_scene_dir_raises(tmp_path: Path) -> None:
    dm = _DM({"data": {"scene_dir": str(tmp_path / "absent")}})
    with pytest.raises(RuntimeError, match="Scene directory not found"):
        dm.setup("fit")


def test_setup_missing_train_split_raises(tmp_path: Path) -> None:
    root = _make_scene_root(tmp_path, splits=("val",))  # no train.txt
    dm = _DM({"data": {"scene_dir": str(root)}})
    with pytest.raises(RuntimeError, match="Missing required split file"):
        dm.setup("fit")


def test_setup_fit_builds_train_and_val(tmp_path: Path) -> None:
    root = _make_scene_root(tmp_path, splits=("train", "val"))
    dm = _DM({"data": {"scene_dir": str(root)}})
    dm.setup("fit")
    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    assert dm.train_dataset is not dm.val_dataset


def test_setup_fit_val_falls_back_to_train(tmp_path: Path) -> None:
    root = _make_scene_root(tmp_path, splits=("train",))  # no val.txt
    dm = _DM({"data": {"scene_dir": str(root)}})
    dm.setup("fit")
    assert dm.val_dataset is dm.train_dataset


def test_setup_test_missing_split_raises(tmp_path: Path) -> None:
    root = _make_scene_root(tmp_path, splits=("train",))  # no test.txt
    dm = _DM({"data": {"scene_dir": str(root)}})
    with pytest.raises(RuntimeError, match="Missing required split file"):
        dm.setup("test")


def test_dataloaders_require_setup(tmp_path: Path) -> None:
    dm = _DM({"data": {"scene_dir": str(tmp_path)}})
    with pytest.raises(RuntimeError, match="Call setup"):
        dm.train_dataloader()
    with pytest.raises(RuntimeError, match="Call setup"):
        dm.val_dataloader()
    with pytest.raises(RuntimeError, match="Call setup"):
        dm.test_dataloader()


def test_train_loader_shuffles_and_drops_last(tmp_path: Path) -> None:
    root = _make_scene_root(tmp_path, splits=("train", "val"))
    dm = _DM({"data": {"scene_dir": str(root), "batch_size": 2, "num_workers": 0, "pin_memory": False}})
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    assert train_loader.drop_last is True
    assert val_loader.drop_last is False
    assert train_loader.batch_size == 2
