"""Unit tests for court detection DataModule test-split wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from src.tasks.court_detection.data import datamodule as dm_mod
from src.tasks.court_detection.data.datamodule import CourtDetectionDataModule

pytestmark = pytest.mark.unit


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
    datamodule = CourtDetectionDataModule(
        OmegaConf.create(
            {
            "data": {
                "task": "kp",
                "data_dir": str(tmp_path),
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": False,
            }
            }
        )
    )

    datamodule.setup("test")

    assert len(created) == 1
    assert created[0]["root"] == tmp_path
    assert created[0]["split"] == "val"
    assert created[0]["is_train"] is False
    assert datamodule.test_dataset is not None
