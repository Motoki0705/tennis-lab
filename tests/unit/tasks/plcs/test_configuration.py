"""PLCS strict configuration contract tests."""

from __future__ import annotations

from copy import deepcopy

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, open_dict

from src.tasks.plcs.configuration import (
    PLCSAnalysisRuntimeConfig,
    PLCSTrainingConfig,
)
from src.tasks.plcs.generate_dataset.config import PLCSGenerationConfig
from src.utils.configuration import (
    MissingConfigurationKeyError,
    PathContractError,
    PathRole,
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)
from src.utils.paths import PROJECT_ROOT


def _training_config() -> DictConfig:
    config_dir = PROJECT_ROOT / "src/tasks/plcs/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name="train")


def _config(config_name: str) -> DictConfig:
    config_dir = PROJECT_ROOT / "src/tasks/plcs/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name=config_name)


def test_training_rejects_legacy_output_root_prefix() -> None:
    config = deepcopy(_training_config())
    with open_dict(config):
        config.run.output_dir = "outputs/legacy"

    with pytest.raises(PathContractError, match="root-prefixed or legacy fragment"):
        PLCSTrainingConfig.from_config(config)


def test_generation_output_is_explicitly_data_root_relative() -> None:
    runtime = PLCSGenerationConfig.from_config(_config("generate_dataset"))

    assert runtime.OUTPUT_ROLE is PathRole.DATA
    assert runtime.output_dir == PROJECT_ROOT / "data/plcs/single_object"


def test_analysis_output_is_explicitly_output_root_relative() -> None:
    runtime = PLCSAnalysisRuntimeConfig.angle_velocity(
        _config("analyze_angle_velocity")
    )

    assert runtime.OUTPUT_ROLE is PathRole.OUTPUT
    assert runtime.output_dir == PROJECT_ROOT / "outputs/plcs/analysis/angle_velocity"
    assert runtime.result_path is not None
    assert runtime.result_path.parent == runtime.output_dir


def test_frame_model_accepts_explicit_sequence_data_profile() -> None:
    config_dir = PROJECT_ROOT / "src/tasks/plcs/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(
            config_name="train",
            overrides=[
                "model=frame",
                "data=singleview_sequence",
                "loss=no_canonical",
            ],
        )

    runtime = PLCSTrainingConfig.from_config(config)
    assert runtime.data.values["mode"] == "sequence"
    assert runtime.data.values["seq_stride"] == 128


def test_tracking_rejects_legacy_invisible_attention_config() -> None:
    config = deepcopy(_config("train_tracking"))
    with open_dict(config.model):
        config.model.mask_invisible_observations = True

    with pytest.raises(
        UnknownConfigurationKeyError, match="mask_invisible_observations"
    ):
        PLCSTrainingConfig.from_config(config)


def test_tracking_requires_positive_four_stage_cycle() -> None:
    config = deepcopy(_config("train_tracking"))
    config.model.num_stages = 2

    with pytest.raises(SemanticConfigurationError, match="positive multiple of 4"):
        PLCSTrainingConfig.from_config(config)


@pytest.mark.parametrize("nested_key", ["mhc", "cswa"])
def test_tracking_requires_nested_architecture_config(nested_key: str) -> None:
    config = deepcopy(_config("train_tracking"))
    with open_dict(config.model):
        del config.model[nested_key]

    with pytest.raises(MissingConfigurationKeyError, match=f"model.{nested_key}"):
        PLCSTrainingConfig.from_config(config)


@pytest.mark.parametrize("nested_key", ["mhc", "cswa"])
def test_tracking_rejects_unknown_nested_architecture_config(
    nested_key: str,
) -> None:
    config = deepcopy(_config("train_tracking"))
    with open_dict(config.model[nested_key]):
        config.model[nested_key].legacy_fallback = True

    with pytest.raises(
        UnknownConfigurationKeyError,
        match=f"model.{nested_key}.legacy_fallback",
    ):
        PLCSTrainingConfig.from_config(config)


def test_tracking_rejects_unknown_cswa_backend() -> None:
    config = deepcopy(_config("train_tracking"))
    config.model.cswa.backend = "auto"

    with pytest.raises(SemanticConfigurationError, match="reference.*cuda"):
        PLCSTrainingConfig.from_config(config)
