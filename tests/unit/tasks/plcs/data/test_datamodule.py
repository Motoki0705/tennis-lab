"""Seed-plumbing tests for the ordinary PLCS DataModule."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.tasks.plcs.data.datamodule import PLCSDataModule

pytestmark = pytest.mark.unit

_CONFIG_DIR = Path("src/tasks/plcs/configs").resolve()


def test_ordinary_datamodule_passes_distinct_replayable_split_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="train")
    received: list[dict[str, object]] = []

    def _dataset(**kwargs: object) -> object:
        received.append(dict(kwargs))
        return object()

    monkeypatch.setattr("src.tasks.plcs.data.datamodule.SceneDataset", _dataset)
    first = PLCSDataModule(config)
    replay = PLCSDataModule(config)

    first._build_dataset(Path("dataset"), "train.txt", True)
    first._build_dataset(Path("dataset"), "val.txt", False)
    replay._build_dataset(Path("dataset"), "train.txt", True)

    assert received[0]["seed"] == received[2]["seed"]
    assert received[0]["seed"] != received[1]["seed"]
