"""Semantic constraints for the DINO SSL execution boundary."""

from __future__ import annotations

from copy import deepcopy

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict
from omegaconf.errors import InterpolationKeyError

from src.tasks.ball_detection.configuration import (
    BallRuntimePaths,
    validate_eval,
    validate_training,
    validate_visualization,
    validate_youtube_boundary,
)
from src.utils.configuration import ConfigurationError
from src.utils.paths import PROJECT_ROOT


def _config() -> DictConfig:
    config_dir = PROJECT_ROOT / "src/tasks/ball_detection/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name="prepare_dinov3_ssl_images")


def _compose(config_name: str, *, overrides: list[str] | None = None) -> DictConfig:
    config_dir = PROJECT_ROOT / "src/tasks/ball_detection/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name=config_name, overrides=overrides or [])


@pytest.mark.parametrize(
    ("path", "invalid"),
    [
        ("workflow.discovery.queries", []),
        ("workflow.discovery.queries", [" "]),
        ("workflow.discovery.max_results_per_query", 0),
        ("workflow.discovery.min_duration_sec", -1),
        ("workflow.discovery.max_duration_sec", -1),
        ("workflow.discovery.min_duration_sec", 4000),
        ("workflow.processing.max_new_videos", 0),
        ("workflow.storage.max_root_gb", 0),
        ("workflow.frames.frames_per_video", 0),
        ("workflow.frames.output_ext", "gif"),
        ("workflow.frames.jpeg_quality", 0),
        ("workflow.frames.jpeg_quality", 101),
        ("workflow.gate.backend", "legacy"),
        ("workflow.gate.vllm.base_url", " "),
        ("workflow.gate.vllm.model", ""),
        ("workflow.gate.vllm.timeout_sec", 0),
        ("workflow.gate.vllm.max_tokens", 0),
        ("workflow.gate.vllm.accept_labels", []),
        ("workflow.gate.vllm.prompt", ""),
        ("workflow.gate.vllm.server.command", []),
        ("workflow.gate.vllm.server.health_url", ""),
        ("workflow.gate.vllm.server.startup_timeout_sec", 0),
        ("workflow.gate.vllm.server.poll_interval_sec", 0),
        ("workflow.gate.vllm.server.request_timeout_sec", 0),
        ("workflow.gate.vllm.server.shutdown_timeout_sec", 0),
    ],
)
def test_dino_ssl_rejects_invalid_semantic_boundary_values(
    path: str,
    invalid: object,
) -> None:
    config = deepcopy(_config())
    with open_dict(config):
        OmegaConf.update(config, path, invalid, merge=False)

    with pytest.raises(ConfigurationError):
        validate_youtube_boundary(config)


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


def test_web_training_rejects_removed_temporal_only_key() -> None:
    config = _compose("train", overrides=["data=web_frames"])
    with open_dict(config):
        config.data.temporal_only = True

    with pytest.raises(ConfigurationError):
        validate_training(config)


def test_derived_output_rejects_parent_escape() -> None:
    paths = BallRuntimePaths.from_config(_compose("train"))

    with pytest.raises(ConfigurationError):
        paths.output("../escape")


def test_visualization_rejects_absolute_clip_path() -> None:
    config = _compose("visualize")
    config.visualization.clip_dir = "/tmp/clip"

    with pytest.raises(ConfigurationError):
        validate_visualization(config)


@pytest.mark.parametrize("field", ["source_id", "url", "split"])
def test_youtube_source_rejects_empty_required_fields(field: str) -> None:
    config = _compose("prepare_youtube_dataset")
    config.workflow.sources[0][field] = ""

    with pytest.raises(ConfigurationError):
        validate_youtube_boundary(config)
