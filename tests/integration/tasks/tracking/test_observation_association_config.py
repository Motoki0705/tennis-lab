"""Cross-task composition tests for strict observation association settings."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from src.tasks.blcs.configuration import validate_training_boundary
from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.utils.configuration import (
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)

pytestmark = pytest.mark.integration

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_EXPECTED_ASSOCIATION: dict[str, dict[str, object]] = {
    "blcs": {
        "max_distance": 0.04,
        "max_missed_frames": 2,
        "min_reuse_gap_frames": 4,
        "use_velocity_prediction": True,
        "min_common_keypoints": 1,
        "cost_reduction": "mean",
        "overflow_policy": "error",
    },
    "plcs": {
        "max_distance": 0.08,
        "max_missed_frames": 8,
        "min_reuse_gap_frames": 4,
        "use_velocity_prediction": True,
        "min_common_keypoints": 4,
        "cost_reduction": "median",
        "overflow_policy": "error",
    },
}


def _compose_tracking_config(task: str, config_name: str) -> DictConfig:
    config_dir = _REPOSITORY_ROOT / "src" / "tasks" / task / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(config_name=config_name)


def _validate_public_boundary(task: str, config: DictConfig) -> object:
    if task == "blcs":
        return validate_training_boundary(config)
    if task == "plcs":
        return PLCSTrainingConfig.from_config(config)
    raise AssertionError(f"Unsupported task: {task!r}.")


@pytest.mark.parametrize(
    ("task", "config_name"),
    [
        ("blcs", "train_tracking"),
        ("blcs", "train_tracking_chunked"),
        ("plcs", "train_tracking"),
        ("plcs", "train_tracking_chunked"),
    ],
)
def test_tracking_configs_compose_exact_task_association(
    task: str,
    config_name: str,
) -> None:
    config = _compose_tracking_config(task, config_name)

    _validate_public_boundary(task, config)
    association = OmegaConf.to_container(config.data.association, resolve=True)
    assert association == _EXPECTED_ASSOCIATION[task]
    assert "randomize_slots_train" not in config.data.lifecycle


@pytest.mark.parametrize("task", ["blcs", "plcs"])
def test_public_boundaries_reject_legacy_and_unknown_association_keys(
    task: str,
) -> None:
    config = _compose_tracking_config(task, "train_tracking")

    legacy = deepcopy(config)
    with open_dict(legacy.data.lifecycle):
        legacy.data.lifecycle.randomize_slots_train = True
    with pytest.raises(UnknownConfigurationKeyError, match="randomize_slots_train"):
        _validate_public_boundary(task, legacy)

    unknown = deepcopy(config)
    with open_dict(unknown.data.association):
        unknown.data.association.legacy_fallback = True
    with pytest.raises(UnknownConfigurationKeyError, match="legacy_fallback"):
        _validate_public_boundary(task, unknown)


@pytest.mark.parametrize(
    ("task", "key", "value", "message"),
    [
        ("blcs", "max_distance", 0.0, "max_distance"),
        ("blcs", "min_common_keypoints", 2, "min_common_keypoints"),
        ("plcs", "max_distance", 0.0, "max_distance"),
        ("plcs", "cost_reduction", "mean", "cost_reduction"),
    ],
)
def test_public_boundaries_reject_invalid_task_association(
    task: str,
    key: str,
    value: Any,
    message: str,
) -> None:
    config = _compose_tracking_config(task, "train_tracking")
    config.data.association[key] = value

    with pytest.raises(SemanticConfigurationError, match=message):
        _validate_public_boundary(task, config)
