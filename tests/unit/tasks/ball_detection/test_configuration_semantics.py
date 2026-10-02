"""Strict configuration for store-only ball detection."""

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, open_dict
from omegaconf.errors import InterpolationKeyError

from src.tasks.ball_detection.configuration import (
    BallRuntimePaths,
    validate_eval,
    validate_training,
    validate_visualization,
)
from src.tasks.ball_detection.data import build_ball_detection_datamodule
from src.utils.configuration import ConfigurationError
from src.utils.paths import PROJECT_ROOT


def _compose(config_name: str, *, overrides: list[str] | None = None) -> DictConfig:
    config_dir = PROJECT_ROOT / "src/tasks/ball_detection/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name=config_name, overrides=overrides or [])


@pytest.mark.parametrize("config_name", ["train", "train_meiji_mixed"])
def test_training_defaults_to_v2_store(config_name: str) -> None:
    config = _compose(config_name)
    validate_training(config)
    assert config.data.source == "store"
    assert config.data.data_dir == "ball_detection/ball-mix-v2"
    assert list(config.data.sources) == ["tracknet", "meiji", "chat_annotation"]


@pytest.mark.parametrize("source", ["web", "staged", "tracknet", "youtube", "mixed_tracknet"])
def test_removed_sources_are_rejected_before_opening_data(source: str) -> None:
    config = _compose("train")
    config.data.source = source
    with pytest.raises(ConfigurationError, match="expected 'store'"):
        validate_training(config)
    with pytest.raises(ConfigurationError, match="expected 'store'"):
        build_ball_detection_datamodule(config)


@pytest.mark.parametrize("section, key", [("data", "t_max"), ("training", "staged")])
def test_variable_length_training_settings_are_rejected(section: str, key: str) -> None:
    config = _compose("train")
    with open_dict(config):
        config[section][key] = 8 if key == "t_max" else {}
    with pytest.raises(ConfigurationError):
        validate_training(config)


@pytest.mark.parametrize(
    "mutation", ["unknown_model", "missing_model", "wrong_type", "run_typo"]
)
def test_training_rejects_invalid_exact_configuration(mutation: str) -> None:
    config = _compose("train")
    with open_dict(config):
        if mutation == "unknown_model":
            config.model["num_frmaes"] = 8
        elif mutation == "missing_model":
            del config.model.num_frames
        elif mutation == "wrong_type":
            config.model.num_frames = "8"
        else:
            config.run["num_frmaes"] = 8

    with pytest.raises((ConfigurationError, InterpolationKeyError)):
        validate_training(config)


def test_eval_rejects_missing_required_field() -> None:
    config = _compose("eval")
    with open_dict(config):
        del config.evaluation.max_batches_per_split

    with pytest.raises(ConfigurationError):
        validate_eval(config)


def test_training_rejects_conflicting_checkpoint_inputs() -> None:
    config = _compose("train")
    config.run.resume = "resume.ckpt"
    config.run.init_weights = "init.ckpt"

    with pytest.raises(ConfigurationError):
        validate_training(config)


def test_derived_output_rejects_parent_escape() -> None:
    paths = BallRuntimePaths.from_config(_compose("train"))

    with pytest.raises(ConfigurationError):
        paths.output("../escape")


def test_visualization_rejects_absolute_clip_path() -> None:
    config = _compose("visualize")
    config.visualization.store_dir = "/tmp/clip"

    with pytest.raises(ConfigurationError):
        validate_visualization(config)
