"""Strict configuration for store-only ball detection."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict
from omegaconf.errors import InterpolationKeyError

from src.tasks.ball_detection.configuration import (
    BallRuntimePaths,
    validate_eval,
    validate_training,
    validate_visualization,
)
from src.tasks.ball_detection.data import build_ball_detection_datamodule
from src.tasks.ball_detection.evaluation.contracts import load_evaluation_manifest
from src.tasks.base.configuration import BaseRunConfig
from src.utils.configuration import ConfigurationError, PathRole
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


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("training.checkpoint.enabled", False),
        ("training.checkpoint.save_top_k", 1),
        ("training.checkpoint.filename", "best"),
        ("training.trainer.check_val_every_n_epoch", 2),
        ("training.validation_candidates.max_candidates", 1),
        ("training.validation_candidates.nms_kernel", 3),
        ("training.validation_candidates.patch_size", 3),
        ("training.validation_candidates.subpixel_refine", False),
        ("training.validation_candidates.radius_source_px", 4.0),
    ],
)
def test_training_rejects_policy_that_loses_epoch_candidates(key: str, value: object) -> None:
    from src.tasks.ball_detection.training.runner import BallDetectionTrainingRunner

    config = _compose("train")
    OmegaConf.update(config, key, value)
    with pytest.raises(ConfigurationError):
        validate_training(config)
    with pytest.raises(ConfigurationError):
        BallDetectionTrainingRunner().validate_runtime_config(config)


def test_training_does_not_default_missing_candidate_settings() -> None:
    config = _compose("train")
    with open_dict(config):
        del config.training.validation_candidates
    with pytest.raises(ConfigurationError, match="validation_candidates"):
        validate_training(config)


@pytest.mark.parametrize(
    ("name", "overrides"),
    [("train", []), ("train", ["training=gan"]), ("train", ["training=lora"]),
     ("train_meiji_mixed", [])],
)
def test_all_training_profiles_keep_every_epoch_with_the_same_candidate_metric(
    name: str, overrides: list[str],
) -> None:
    from src.tasks.ball_detection.configuration import validate_epoch_candidate_policy

    config = _compose(name, overrides=overrides)
    validate_epoch_candidate_policy(config)
    assert config.training.checkpoint.save_top_k == -1
    assert config.training.checkpoint.mode == "max"
    assert config.training.checkpoint.monitor == (
        "val/meiji/candidate_recall_at_8_20px" if name == "train_meiji_mixed" else "val/candidate_recall_at_8_20px"
    )


def test_derived_output_rejects_parent_escape() -> None:
    paths = BallRuntimePaths.from_config(_compose("train"))

    with pytest.raises(ConfigurationError):
        paths.output("../escape")


def test_visualization_rejects_absolute_clip_path() -> None:
    config = _compose("visualize")
    config.visualization.store_dir = "/tmp/clip"

    with pytest.raises(ConfigurationError):
        validate_visualization(config)


@pytest.mark.parametrize("initialize_from_run", [False, True])
def test_training_keeps_backbone_assets_and_initial_weights_in_separate_roots(
    tmp_path: Path, initialize_from_run: bool,
) -> None:
    overrides = [
        f"paths.project_root={tmp_path}",
        f"paths.artifact_root={tmp_path / 'previous-runs'}",
        "model=dinov3_rope",
    ]
    if initialize_from_run:
        overrides.append("run.init_weights={role:artifact,path:ball_detection/train/previous/checkpoints/last.ckpt}")
    config = _compose("train", overrides=overrides)
    validate_training(config)
    paths = BallRuntimePaths.from_config(config)
    assert paths.checkpoint(str(config.model.backbone.checkpoint_path)) == (
        tmp_path / "ckpt/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    )
    run = BaseRunConfig.from_mapping(config.run, resolver=paths.resolver)
    if initialize_from_run:
        assert run.init_weights == tmp_path / "previous-runs/ball_detection/train/previous/checkpoints/last.ckpt"
    else:
        assert run.init_weights is None


@pytest.mark.parametrize("name", ["eval", "visualize", "evaluate_manifest"])
def test_historical_checkpoint_inputs_preserve_the_independent_backbone_root(tmp_path: Path, name: str) -> None:
    config = _compose(name, overrides=[f"paths.project_root={tmp_path}", f"paths.artifact_root={tmp_path / 'previous-runs'}"])
    paths = BallRuntimePaths.from_config(config)
    assert paths.resolver.roots.checkpoint_root == tmp_path / "ckpt"
    if name == "evaluate_manifest":
        manifest = load_evaluation_manifest(PROJECT_ROOT / config.manifest_path, resolver=paths.resolver)
        assert all(model.checkpoint.is_relative_to(tmp_path / "previous-runs") for model in manifest.models)
    else:
        if name == "eval":
            section, key, location = config.run, "checkpoint_path", "run"
        else:
            section, key, location = config.visualization, "checkpoint", "visualization"
        reference = paths.checkpoint_input(section, key, path=location)
        assert reference.role is PathRole.ARTIFACT
        assert reference.path.is_relative_to(tmp_path / "previous-runs")
