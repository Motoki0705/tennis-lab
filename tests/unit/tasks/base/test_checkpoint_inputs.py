"""Training checkpoint sources remain explicit when assets and runs use different roots."""

from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import src.utils.hydra  # noqa: F401 -- public output/run-id resolvers
from src.tasks.ball_detection.configuration import BallRuntimePaths, validate_training
from src.tasks.base.configuration import (
    BaseRunConfig,
    TrainingRuntimeConfig,
    resolve_checkpoint_input,
)
from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.court_detection.configuration import CourtTrainingConfig
from src.utils.configuration import ConfigurationError, PathRole
from src.utils.paths import PROJECT_ROOT


@pytest.mark.parametrize("field", ["resume", "init_weights"])
@pytest.mark.parametrize("role", [None, "checkpoint", "artifact"])
def test_training_inputs_resolve_from_only_the_declared_root(
    tmp_path: Path, make_training_config: Any, field: str, role: str | None,
) -> None:
    fragment = "prior/run/last.ckpt"
    value = fragment if role is None else {"role": role, "path": fragment}
    config = make_training_config(run={field: value})
    config["paths"].update(checkpoint_root="weights", artifact_root="previous-runs", output_root="new-runs")
    for name in ("weights", "previous-runs"):
        source = tmp_path / name / fragment
        source.parent.mkdir(parents=True)
        source.write_bytes(name.encode())
    runtime = TrainingRuntimeConfig.from_config(OmegaConf.create(config), repository_root=tmp_path)
    expected = tmp_path / ("previous-runs" if role == "artifact" else "weights") / fragment
    assert getattr(runtime.run, field) == expected
    assert runtime.run.output_dir == tmp_path / "new-runs/run"
    if field == "resume":
        assert BaseTrainingRunner().resolve_resume(runtime, runtime.run.output_dir) == str(expected)


@pytest.mark.parametrize("field", ["resume", "init_weights"])
@pytest.mark.parametrize("value", [
    "", "/tmp/outside.ckpt", "../outside.ckpt",
    {}, {"role": "artifact"}, {"path": "run/model.ckpt"},
    {"role": "artifact", "path": "run/model.ckpt", "fallback": "checkpoint"},
    {"role": "output", "path": "run/model.ckpt"},
    {"role": "data", "path": "run/model.ckpt"},
    {"role": "typo", "path": "run/model.ckpt"},
    {"role": None, "path": "run/model.ckpt"},
    {"role": "artifact", "path": None},
    {"role": "artifact", "path": ""},
    {"role": "artifact", "path": "/tmp/outside.ckpt"},
    {"role": "checkpoint", "path": "../outside.ckpt"},
    {"role": "artifact", "path": "../outside.ckpt"},
    {"role": "artifact", "path": "nested/../../outside.ckpt"},
])
def test_invalid_training_checkpoint_declarations_fail_before_loading(
    tmp_path: Path, make_training_config: Any, field: str, value: object,
) -> None:
    config = make_training_config(run={field: value})
    with pytest.raises(ConfigurationError):
        TrainingRuntimeConfig.from_config(OmegaConf.create(config), repository_root=tmp_path)


@pytest.mark.parametrize("field", ["resume", "init_weights"])
def test_declared_artifact_cannot_follow_a_symlink_outside_its_root(
    tmp_path: Path, make_training_config: Any, field: str,
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    outside = tmp_path / "outside.ckpt"
    outside.touch()
    (artifacts / "escaped.ckpt").symlink_to(outside)
    config = make_training_config(run={field: {"role": "artifact", "path": "escaped.ckpt"}})
    with pytest.raises(ConfigurationError):
        TrainingRuntimeConfig.from_config(config, repository_root=tmp_path)


def test_distinct_roles_do_not_allow_resume_and_initial_weights_together(
    tmp_path: Path, make_training_config: Any,
) -> None:
    config = make_training_config(run={
        "resume": {"role": "artifact", "path": "run/last.ckpt"},
        "init_weights": {"role": "checkpoint", "path": "adopted.ckpt"},
    })
    with pytest.raises(ConfigurationError, match="mutually exclusive"):
        TrainingRuntimeConfig.from_config(config, repository_root=tmp_path)


@pytest.mark.parametrize("task", ["ball_detection", "court_detection"])
@pytest.mark.parametrize("field", ["resume", "init_weights"])
def test_task_hydra_training_boundaries_accept_explicit_artifact_inputs(
    tmp_path: Path, task: str, field: str,
) -> None:
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "src/tasks" / task / "configs")):
        config = compose(config_name="train", overrides=[
            f"paths.checkpoint_root={tmp_path / 'weights'}",
            f"paths.artifact_root={tmp_path / 'previous-runs'}",
            f"run.{field}={{role:artifact,path:prior/run/last.ckpt}}",
        ])
    if task == "court_detection":
        run = CourtTrainingConfig.from_config(config).shared.run
    else:
        validate_training(config)
        run = BaseRunConfig.from_mapping(config.run, resolver=BallRuntimePaths.from_config(config).resolver)
    assert getattr(run, field) == tmp_path / "previous-runs/prior/run/last.ckpt"


def test_model_consumer_retains_artifact_authority_without_rebinding_pretrained_root(
    tmp_path: Path, make_training_config: Any,
) -> None:
    config = make_training_config()
    config["paths"].update(checkpoint_root="weights", artifact_root="prior-runs")
    resolver = TrainingRuntimeConfig.from_config(config, repository_root=tmp_path).resolver
    for name in ("weights", "prior-runs"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "model.ckpt").write_text(name)
    reference = resolve_checkpoint_input(
        {"model": {"role": "artifact", "path": "model.ckpt"}}, "model", path="evaluation", resolver=resolver,
    )
    assert reference is not None and reference.role is PathRole.ARTIFACT
    checked = BasePredictor._ensure_checkpoint(reference.path, resolver=resolver, role=reference.role)
    assert checked == [tmp_path / "prior-runs/model.ckpt"]
    assert resolver.resolve(PathRole.CHECKPOINT, "backbone.pth") == tmp_path / "weights/backbone.pth"
    reference.path.unlink()
    with pytest.raises(FileNotFoundError):
        BasePredictor._ensure_checkpoint(reference.path, resolver=resolver, role=reference.role)
    with pytest.raises(ConfigurationError):
        BasePredictor._ensure_checkpoint(tmp_path / "weights/model.ckpt", resolver=resolver, role=PathRole.ARTIFACT)
    with pytest.raises(ValueError, match="authority"):
        BasePredictor._ensure_checkpoint("model.ckpt", resolver=resolver, role=PathRole.DATA)


def test_public_checkpoint_reference_keeps_legacy_string_and_null_semantics(
    tmp_path: Path, make_training_config: Any,
) -> None:
    resolver = TrainingRuntimeConfig.from_config(make_training_config(), repository_root=tmp_path).resolver
    reference = resolve_checkpoint_input({"model": "selected.ckpt"}, "model", path="inference", resolver=resolver)
    assert reference is not None and reference.role is PathRole.CHECKPOINT
    assert reference.path == tmp_path / "selected.ckpt"
    assert resolve_checkpoint_input({"model": None}, "model", path="inference", resolver=resolver) is None
