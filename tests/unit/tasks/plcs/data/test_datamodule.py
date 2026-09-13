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


def test_train_weights_increase_hard_sampling_and_are_replayable(
    tmp_path: Path,
) -> None:
    import json

    from src.tasks.plcs.data.dataset import SceneDataset

    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train", overrides=["+data.sampling_weights=weights.json"]
        )
    scenes = [tmp_path / f"scene_{i}" for i in range(1000)]
    weights = {p.name: (3.0 if i < 200 else 0.5) for i, p in enumerate(scenes)}
    (tmp_path / "weights.json").write_text(json.dumps(weights))

    def sampler_indices() -> list[int]:
        dm = PLCSDataModule(config)
        dm.scene_dir = tmp_path
        dataset = object.__new__(SceneDataset)
        dataset.scenes = scenes
        dm.train_dataset = dataset
        return list(dm.train_dataloader().sampler)

    first = sampler_indices()
    assert first == sampler_indices()
    assert 0.55 < sum(i < 200 for i in first) / len(first) < 0.65
    weights["outside_train"] = 1.0
    (tmp_path / "weights.json").write_text(json.dumps(weights))
    with pytest.raises(ValueError, match="complete filtered train split"):
        sampler_indices()
    weights.pop("outside_train")
    weights[scenes[0].name] = float("nan")
    (tmp_path / "weights.json").write_text(json.dumps(weights))
    with pytest.raises(ValueError, match="finite and positive"):
        sampler_indices()
