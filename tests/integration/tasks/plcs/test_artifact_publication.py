"""PLCS standalone generation and training publication integration tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, open_dict

import src.tasks.plcs.scripts.generate_dataset as generate_dataset_script
from src.tasks.base.data.court_coordinate_contract import (
    COURT_COORDINATE_NORMALIZATION_METADATA_KEY,
)
from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.generate_dataset.config import PLCSGenerationConfig
from src.tasks.plcs.training.runner import PLCSTrainingRunner
from src.utils.paths import PROJECT_ROOT


def _config(config_name: str) -> DictConfig:
    config_dir = PROJECT_ROOT / "src/tasks/plcs/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name=config_name)


def _isolated_config(config_name: str, tmp_path: Path) -> DictConfig:
    config = _config(config_name)
    project_root = tmp_path / "project"
    with open_dict(config.paths):
        config.paths.project_root = str(project_root)
        config.paths.data_root = "published-data"
        config.paths.checkpoint_root = "published-checkpoints"
        config.paths.artifact_root = "published-artifacts"
        config.paths.output_root = "published-outputs"
        config.paths.cache_root = "published-cache"
        config.paths.external_asset_root = "published-assets"
    return config


def _run_generate_dataset(config: DictConfig) -> int:
    wrapped_attribute = "__wrapped__"
    wrapped = cast(
        Callable[[DictConfig], int],
        getattr(generate_dataset_script.main, wrapped_attribute),
    )
    return wrapped(config)


def _tree_bytes(root: Path) -> dict[str, bytes | None]:
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in sorted(root.rglob("*"))
    }


def test_generate_dataset_helper_fails_loudly_without_hydra_wrapped_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generate_dataset_script, "main", lambda: 0)

    with pytest.raises(AttributeError, match="__wrapped__"):
        _run_generate_dataset(DictConfig({}))


@pytest.mark.parametrize(
    ("config_name", "output_relative", "precreate_empty", "expected_version"),
    [
        ("generate_dataset_norm_v2", "first_norm_v2", False, "v2"),
        ("generate_dataset_norm_v2", "second_norm_v2", True, "v2"),
        ("generate_dataset", "legacy-compatible", False, "v1"),
    ],
)
def test_standalone_generation_publishes_missing_or_empty_versioned_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config_name: str,
    output_relative: str,
    precreate_empty: bool,
    expected_version: str,
) -> None:
    config = _isolated_config(config_name, tmp_path)
    config.run.output_dir = output_relative
    runtime = PLCSGenerationConfig.from_config(config)
    if precreate_empty:
        runtime.output_dir.mkdir(parents=True)
    monkeypatch.setattr(
        generate_dataset_script,
        "generate_parallel_scenes",
        lambda **_kwargs: [],
    )

    result = _run_generate_dataset(config)

    assert result == 0
    assert (runtime.output_dir / "config.yaml").is_file()
    root_metadata = json.loads(
        (runtime.output_dir / "meta.json").read_text(encoding="utf-8")
    )
    contract_metadata = root_metadata[COURT_COORDINATE_NORMALIZATION_METADATA_KEY]
    assert contract_metadata["version"] == expected_version
    if expected_version == "v2":
        assert "norm_v2" in runtime.output_dir.name
        assert contract_metadata["scale_xyz"] == [11.885, 11.885, 11.885]


def test_standalone_generation_preserves_occupied_root_before_config_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _isolated_config("generate_dataset_norm_v2", tmp_path)
    config.run.output_dir = "occupied_norm_v2"
    runtime = PLCSGenerationConfig.from_config(config)
    runtime.output_dir.mkdir(parents=True)
    config_path = runtime.output_dir / "config.yaml"
    config_path.write_bytes(b"legacy-config-must-survive\x00")
    scene_sentinel = runtime.output_dir / "scenes" / "legacy" / "position.npy"
    scene_sentinel.parent.mkdir(parents=True)
    scene_sentinel.write_bytes(b"legacy-scene-must-survive\xff")
    before = _tree_bytes(runtime.output_dir)
    monkeypatch.setattr(
        generate_dataset_script,
        "generate_parallel_scenes",
        lambda **_kwargs: pytest.fail("generation ran for an occupied destination"),
    )

    with pytest.raises(FileExistsError, match="non-empty or non-directory"):
        _run_generate_dataset(config)

    assert _tree_bytes(runtime.output_dir) == before
    assert config_path.read_bytes() == b"legacy-config-must-survive\x00"
    assert scene_sentinel.read_bytes() == b"legacy-scene-must-survive\xff"


@pytest.mark.parametrize("destination_kind", ["directory", "file"])
def test_v2_training_rejects_occupied_output_before_writes_and_preserves_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    destination_kind: str,
) -> None:
    config = _isolated_config("train_norm_v2", tmp_path)
    config.run.output_dir = "plcs/occupied_norm_v2"
    runtime = PLCSTrainingConfig.from_config(config)
    output_dir = runtime.shared.run.output_dir
    if destination_kind == "directory":
        output_dir.mkdir(parents=True)
        sentinel = output_dir / "last.ckpt"
        sentinel.write_bytes(b"existing-v2-checkpoint\x00\xff")
        before = _tree_bytes(output_dir)
    else:
        output_dir.parent.mkdir(parents=True)
        output_dir.write_bytes(b"existing-non-directory-output")
        before = {}

    runner = PLCSTrainingRunner()
    monkeypatch.setattr(
        runner,
        "seed_everything",
        lambda _runtime: pytest.fail("training continued past occupancy validation"),
    )

    with pytest.raises(FileExistsError, match="non-empty or non-directory"):
        runner.run(config)

    if destination_kind == "directory":
        assert _tree_bytes(output_dir) == before
        assert sentinel.read_bytes() == b"existing-v2-checkpoint\x00\xff"
    else:
        assert output_dir.read_bytes() == b"existing-non-directory-output"


@pytest.mark.parametrize("precreate_empty", [False, True])
def test_v2_training_accepts_missing_or_empty_output_without_writing_during_validation(
    tmp_path: Path,
    precreate_empty: bool,
) -> None:
    config = _isolated_config("train_norm_v2", tmp_path)
    config.run.output_dir = "plcs/control_norm_v2"
    runtime = PLCSTrainingConfig.from_config(config)
    output_dir = runtime.shared.run.output_dir
    if precreate_empty:
        output_dir.mkdir(parents=True)

    PLCSTrainingRunner().prepare_config(config)

    if precreate_empty:
        assert output_dir.is_dir()
        assert list(output_dir.iterdir()) == []
    else:
        assert not output_dir.exists()


def test_v1_training_keeps_unqualified_occupied_output_compatibility(
    tmp_path: Path,
) -> None:
    config = _isolated_config("train", tmp_path)
    config.run.output_dir = "plcs/legacy-compatible"
    runtime = PLCSTrainingConfig.from_config(config)
    output_dir = runtime.shared.run.output_dir
    output_dir.mkdir(parents=True)
    sentinel = output_dir / "legacy.ckpt"
    sentinel.write_bytes(b"legacy-v1-checkpoint")

    PLCSTrainingRunner().prepare_config(config)

    assert sentinel.read_bytes() == b"legacy-v1-checkpoint"
