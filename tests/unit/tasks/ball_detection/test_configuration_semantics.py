"""Semantic constraints for the DINO SSL execution boundary."""

from __future__ import annotations

from copy import deepcopy
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
    validate_youtube_boundary,
)
from src.tasks.ball_detection.evaluation.contracts import load_evaluation_manifest
from src.tasks.base.configuration import BaseRunConfig
from src.utils.configuration import ConfigurationError, PathRole
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
     ("train_meiji_mixed", []), ("train_staged", []),
     ("staged_phase1", []), ("staged_phase2", []), ("staged_phase3", []), ("staged_phase4", [])],
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
    config.visualization.store_dir = "/tmp/clip"

    with pytest.raises(ConfigurationError):
        validate_visualization(config)


@pytest.mark.parametrize("field", ["source_id", "url", "split"])
def test_youtube_source_rejects_empty_required_fields(field: str) -> None:
    config = _compose("prepare_youtube_dataset")
    config.workflow.sources[0][field] = ""

    with pytest.raises(ConfigurationError):
        validate_youtube_boundary(config)


@pytest.mark.parametrize("name", ["train", "train_staged", "staged_phase1", "staged_phase2", "staged_phase3", "staged_phase4"])
def test_training_keeps_backbone_assets_and_prior_phase_runs_in_separate_roots(tmp_path: Path, name: str) -> None:
    overrides = [f"paths.project_root={tmp_path}", f"paths.artifact_root={tmp_path / 'previous-runs'}"]
    if name == "train":
        overrides.append("model=dinov3_rope")
    config = _compose(name, overrides=overrides)
    validate_training(config)
    paths = BallRuntimePaths.from_config(config)
    assert paths.checkpoint(str(config.model.backbone.checkpoint_path)) == (
        tmp_path / "ckpt/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    )
    run = BaseRunConfig.from_mapping(config.run, resolver=paths.resolver)
    if name in {"staged_phase2", "staged_phase3", "staged_phase4"}:
        previous = int(name[-1]) - 1
        assert run.init_weights == tmp_path / f"previous-runs/ball_detection/train/staged/phase{previous}/logs/run/checkpoints/last.ckpt"
    else:
        assert run.init_weights is None


@pytest.mark.parametrize("name", ["eval", "visualize", "evaluate_manifest", "clip_and_predict_youtube_dataset"])
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
        elif name == "visualize":
            section, key, location = config.visualization, "checkpoint", "visualization"
        else:
            section, key, location = config.workflow.prediction, "checkpoint", "workflow.prediction"
        reference = paths.checkpoint_input(section, key, path=location)
        assert reference.role is PathRole.ARTIFACT
        assert reference.path.is_relative_to(tmp_path / "previous-runs")
