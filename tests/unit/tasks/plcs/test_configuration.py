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
from src.utils.configuration import PathContractError, PathRole
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
    assert runtime.output_dir == PROJECT_ROOT / "data/plcs"


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
