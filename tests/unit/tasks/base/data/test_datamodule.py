"""Unit tests for SceneDirectoryDataModule setup/validation logic."""

from __future__ import annotations

from pathlib import Path

import pytest
from torch.utils.data import Dataset

from src.tasks.base.data.datamodule import SceneDirectoryDataModule
from src.utils.configuration import MissingConfigurationKeyError

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


def _config(
    data_root: Path,
    *,
    scene_dir: str | None = None,
    batch_size: int = 4,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> dict[str, object]:
    role_root = data_root if scene_dir is not None else data_root.parent
    scene_child = scene_dir if scene_dir is not None else data_root.name
    return {
        "paths": {
            "project_root": str(role_root),
            "data_root": str(role_root),
            "checkpoint_root": "checkpoints",
            "artifact_root": "artifacts",
            "output_root": "outputs",
            "cache_root": ".cache",
            "external_asset_root": "external",
        },
        "data": {
            "scene_dir": scene_child,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        },
    }


def test_config_parsing_rejects_missing_paths() -> None:
    with pytest.raises(MissingConfigurationKeyError, match="configuration.paths"):
        _DM({})


def test_config_parsing_explicit_values(tmp_path: Path) -> None:
    dm = _DM(
        _config(
            tmp_path,
            scene_dir="scenes",
            batch_size=8,
            num_workers=2,
            pin_memory=False,
        )
    )
    assert dm.batch_size == 8
    assert dm.num_workers == 2
    assert dm.pin_memory is False
    assert dm.scene_dir == tmp_path / "scenes"


def test_setup_missing_scene_dir_raises(tmp_path: Path) -> None:
    dm = _DM(_config(tmp_path, scene_dir="absent"))
    with pytest.raises(RuntimeError, match="Scene directory not found"):
        dm.setup("fit")


def test_setup_missing_train_split_raises(tmp_path: Path) -> None:
    root = _make_scene_root(tmp_path, splits=("val",))  # no train.txt
    dm = _DM(_config(root))
    with pytest.raises(RuntimeError, match="Missing required split file"):
        dm.setup("fit")


def test_setup_fit_builds_train_and_val(tmp_path: Path) -> None:
    root = _make_scene_root(tmp_path, splits=("train", "val"))
    dm = _DM(_config(root))
    dm.setup("fit")
    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    assert dm.train_dataset is not dm.val_dataset


def test_setup_fit_missing_val_split_raises(tmp_path: Path) -> None:
    root = _make_scene_root(tmp_path, splits=("train",))  # no val.txt
    dm = _DM(_config(root))
    with pytest.raises(RuntimeError, match="Missing required split file"):
        dm.setup("fit")


def test_setup_test_missing_split_raises(tmp_path: Path) -> None:
    root = _make_scene_root(tmp_path, splits=("train",))  # no test.txt
    dm = _DM(_config(root))
    with pytest.raises(RuntimeError, match="Missing required split file"):
        dm.setup("test")


def test_dataloaders_require_setup(tmp_path: Path) -> None:
    dm = _DM(_config(tmp_path))
    with pytest.raises(RuntimeError, match="Call setup"):
        dm.train_dataloader()
    with pytest.raises(RuntimeError, match="Call setup"):
        dm.val_dataloader()
    with pytest.raises(RuntimeError, match="Call setup"):
        dm.test_dataloader()


def test_train_loader_shuffles_and_drops_last(tmp_path: Path) -> None:
    root = _make_scene_root(tmp_path, splits=("train", "val"))
    dm = _DM(_config(root, batch_size=2))
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    assert train_loader.drop_last is True
    assert val_loader.drop_last is False
    assert train_loader.batch_size == 2
