"""Seed-plumbing tests for the ordinary BLCS DataModule."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.tasks.blcs.data.datamodule import BLCSDataModule

pytestmark = pytest.mark.unit

_CONFIG_DIR = Path("src/tasks/blcs/configs").resolve()


def test_ordinary_datamodule_passes_distinct_replayable_split_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="train")
    received: list[dict[str, object]] = []

    def _dataset(**kwargs: object) -> object:
        received.append(dict(kwargs))
        return object()

    monkeypatch.setattr("src.tasks.blcs.data.datamodule.BallTrajectoryDataset", _dataset)
    first = BLCSDataModule(config, collate_fn=lambda batch: batch)
    replay = BLCSDataModule(config, collate_fn=lambda batch: batch)

    first._build_dataset(Path("dataset"), "train.txt", True)
    first._build_dataset(Path("dataset"), "val.txt", False)
    replay._build_dataset(Path("dataset"), "train.txt", True)

    assert received[0]["seed"] == received[2]["seed"]
    assert received[0]["seed"] != received[1]["seed"]
