"""Unit tests for court detection DataModule test-split wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir

from src.tasks.court_detection.data import datamodule as dm_mod
from src.tasks.court_detection.data.datamodule import CourtDetectionDataModule

pytestmark = pytest.mark.unit

_CONFIG_DIR = Path(__file__).resolve().parents[5] / "src/tasks/court_detection/configs"


def test_setup_test_uses_data_val_json_split_for_kp(monkeypatch, tmp_path: Path) -> None:
    created: list[dict[str, Any]] = []

    class _FakeCourtKPDataset:
        def __init__(
            self,
            root: str | Path,
            split: str,
            is_train: bool,
            config: dict[str, Any],
        ) -> None:
            created.append(
                {
                    "root": Path(root),
                    "split": split,
                    "is_train": is_train,
                    "config": config,
                }
            )

        def __len__(self) -> int:
            return 0

    monkeypatch.setattr(dm_mod, "CourtKPDataset", _FakeCourtKPDataset)
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train",
            overrides=["data=court_kp", "loss=kp"],
        )
    config.paths.project_root = str(tmp_path)
    config.paths.data_root = "data"
    config.data.data_dir = "court"
    config.data.batch_size = 2
    config.data.num_workers = 0
    config.data.pin_memory = False
    datamodule = CourtDetectionDataModule(config)

    datamodule.setup("test")

    assert len(created) == 1
    assert created[0]["root"] == tmp_path / "data" / "court"
    assert created[0]["split"] == "val"
    assert created[0]["is_train"] is False
    assert datamodule.test_dataset is not None
