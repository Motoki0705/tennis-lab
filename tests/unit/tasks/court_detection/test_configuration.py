"""Typed and Hydra composition contracts for Court detection configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, open_dict

from src.tasks.court_detection.configuration import (
    CourtTrainingConfig,
    SyntheticCourtSourceConfig,
)
from src.utils.configuration import (
    ConfigurationTypeError,
    MissingConfigurationKeyError,
    SemanticConfigurationError,
)

_CONFIG_DIR = Path(__file__).resolve().parents[4] / "src/tasks/court_detection/configs"


def _compose(source: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="train",
            overrides=[f"data/source={source}", "data/processing=kp"],
        )


@pytest.mark.parametrize(
    ("source", "schema"),
    [("synthetic_court_v1", "v1"), ("synthetic_court", "v2")],
)
def test_hydra_explicitly_composes_each_synthetic_schema(
    source: str,
    schema: str,
) -> None:
    runtime = CourtTrainingConfig.from_config(_compose(source))

    assert isinstance(runtime.data.source, SyntheticCourtSourceConfig)
    assert runtime.data.source.schema == schema
    assert runtime.data.source.kind == "synthetic_court"


def test_synthetic_schema_cannot_be_omitted_or_guessed() -> None:
    missing = deepcopy(_compose("synthetic_court"))
    with open_dict(missing.data.source):
        del missing.data.source.schema
    with pytest.raises(MissingConfigurationKeyError, match="data.source.schema"):
        CourtTrainingConfig.from_config(missing)

    unknown = deepcopy(_compose("synthetic_court"))
    unknown.data.source.schema = "auto"
    with pytest.raises(SemanticConfigurationError, match="explicitly 'v1' or 'v2'"):
        CourtTrainingConfig.from_config(unknown)


@pytest.mark.parametrize("scene_id", [".", ".."])
def test_synthetic_scene_ids_reject_dot_segments(scene_id: str) -> None:
    config = _compose("synthetic_court")
    config.data.source.scene_ids = [scene_id]

    with pytest.raises(ConfigurationTypeError, match="safe non-empty scene IDs"):
        CourtTrainingConfig.from_config(config)


def test_tennis_default_has_no_validation_as_test_mapping() -> None:
    runtime = CourtTrainingConfig.from_config(_compose("tennis_court_detector"))

    assert runtime.data.source.kind == "tennis_court_detector"
    assert runtime.data.source.split_mapping["test"] is None


def test_tennis_rejects_validation_as_test_mapping() -> None:
    config = _compose("tennis_court_detector")
    config.data.source.split_mapping.test = "val"

    with pytest.raises(SemanticConfigurationError, match="cannot be reused as test"):
        CourtTrainingConfig.from_config(config)
