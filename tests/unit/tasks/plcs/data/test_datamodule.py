"""Seed-plumbing and loader-contract tests for the ordinary PLCS DataModule."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    WeightedRandomSampler,
)

from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.data.datamodule import PLCSDataModule
from src.tasks.plcs.data.dataset import SceneDataset
from src.utils.configuration import (
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)

pytestmark = pytest.mark.unit

_CONFIG_DIR = Path("src/tasks/plcs/configs").resolve()


def _compose_train(overrides: Sequence[str] = ()) -> DictConfig:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        return compose(config_name="train", overrides=list(overrides))


def _fake_scene_dataset(scenes: Sequence[Path]) -> SceneDataset:
    """Build a worker-seeded SceneDataset stub without reading real scenes."""
    dataset = object.__new__(SceneDataset)
    dataset.scenes = list(scenes)
    return dataset


def _module_with_fake_datasets(
    config: DictConfig, scene_dir: Path, scenes: Sequence[Path]
) -> PLCSDataModule:
    module = PLCSDataModule(config)
    module.scene_dir = scene_dir
    module.train_dataset = _fake_scene_dataset(scenes)
    module.val_dataset = _fake_scene_dataset(scenes)
    module.test_dataset = _fake_scene_dataset(scenes)
    return module


def _loader_contract(loader: DataLoader) -> dict[str, Any]:
    return {
        "batch_size": loader.batch_size,
        "num_workers": loader.num_workers,
        "pin_memory": loader.pin_memory,
        "drop_last": loader.drop_last,
        "collate_fn": loader.collate_fn,
        "worker_init_fn": loader.worker_init_fn,
    }


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


def test_sampling_weights_keep_the_shared_loader_contract(tmp_path: Path) -> None:
    scenes = [tmp_path / f"scene_{i}" for i in range(9)]
    weights = {path.name: float(index + 1) for index, path in enumerate(scenes)}
    (tmp_path / "weights.json").write_text(json.dumps(weights))

    unweighted = _module_with_fake_datasets(
        _compose_train(["data.batch_size=4"]), tmp_path, scenes
    )
    weighted = _module_with_fake_datasets(
        _compose_train(["+data.sampling_weights=weights.json", "data.batch_size=4"]),
        tmp_path,
        scenes,
    )

    unweighted_train = unweighted.train_dataloader()
    weighted_train = weighted.train_dataloader()
    unweighted_val = unweighted.val_dataloader()
    weighted_val = weighted.val_dataloader()
    unweighted_test = unweighted.test_dataloader()
    weighted_test = weighted.test_dataloader()

    assert isinstance(unweighted_train.sampler, RandomSampler)
    assert isinstance(weighted_train.sampler, WeightedRandomSampler)

    shared_keys = (
        "batch_size",
        "num_workers",
        "pin_memory",
        "collate_fn",
        "worker_init_fn",
    )
    unweighted_contract = _loader_contract(unweighted_train)
    weighted_contract = _loader_contract(weighted_train)
    for key in shared_keys:
        assert weighted_contract[key] == unweighted_contract[key]
    assert unweighted_contract["drop_last"] is True
    assert weighted_contract["drop_last"] is True

    # Weighted sampling preserves one draw per train scene while val/test stay
    # sequential and deterministic.
    assert len(unweighted_train.sampler) == len(scenes)
    assert len(weighted_train.sampler) == len(scenes)
    assert weighted_train.sampler.replacement is True
    assert len(weighted_train) == len(unweighted_train) == 2

    val_contract = _loader_contract(unweighted_val)
    for loader in (unweighted_val, weighted_val, unweighted_test, weighted_test):
        assert isinstance(loader.sampler, SequentialSampler)
        assert len(loader.sampler) == len(scenes)
        assert len(loader) == 3
        contract = _loader_contract(loader)
        for key in shared_keys:
            assert contract[key] == val_contract[key]
        assert contract["drop_last"] is False


def test_sampling_weights_are_accepted_only_for_the_default_backend() -> None:
    runtime = PLCSTrainingConfig.from_config(
        _compose_train(["+data.sampling_weights=weights.json"])
    )
    assert runtime.data.values["sampling_weights"] == "weights.json"


@pytest.mark.parametrize(
    ("config_name", "weights_name", "error", "message"),
    [
        (
            "train_chunked",
            "weights.json",
            SemanticConfigurationError,
            "requires backend=default",
        ),
        (
            "train_tracking",
            "weights.json",
            UnknownConfigurationKeyError,
            r"data\.sampling_weights",
        ),
        (
            "train",
            "../weights.json",
            SemanticConfigurationError,
            "non-empty file name",
        ),
    ],
)
def test_sampling_weights_are_rejected_outside_default_backend(
    config_name: str,
    weights_name: str,
    error: type[Exception],
    message: str,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name=config_name,
            overrides=[f"+data.sampling_weights={weights_name}"],
        )

    with pytest.raises(error, match=message):
        PLCSTrainingConfig.from_config(config)


@pytest.mark.parametrize("bad_weight", [0.0, -1.0, float("inf"), True])
def test_sampling_weights_must_be_finite_positive_numbers(
    tmp_path: Path, bad_weight: object
) -> None:
    scenes = [tmp_path / "scene_0"]
    (tmp_path / "weights.json").write_text(json.dumps({"scene_0": bad_weight}))
    module = _module_with_fake_datasets(
        _compose_train(["+data.sampling_weights=weights.json"]), tmp_path, scenes
    )

    with pytest.raises(ValueError, match="finite and positive"):
        module.train_dataloader()
