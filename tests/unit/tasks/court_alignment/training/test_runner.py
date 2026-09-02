"""Unit tests for court-alignment post-fit checkpoint selection."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pytorch_lightning.callbacks import ModelCheckpoint

from src.tasks.court_alignment.training.runner import CourtAlignmentTrainingRunner
from src.utils.configuration import (
    PathContractError,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)


def _runtime(*, checkpoint_enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        training=SimpleNamespace(checkpoint=SimpleNamespace(enabled=checkpoint_enabled))
    )


def _path_resolver(tmp_path: Path) -> PathResolver:
    return PathResolver(
        RuntimePathRoots.from_mapping(
            {
                "project_root": ".",
                "data_root": "data",
                "checkpoint_root": "checkpoints",
                "artifact_root": "artifacts",
                "output_root": "outputs",
                "cache_root": "cache-files",
                "external_asset_root": "external-files",
            },
            repository_root=tmp_path.resolve(),
        )
    )


def _install_evaluation_runtime(
    runner: CourtAlignmentTrainingRunner,
    monkeypatch: pytest.MonkeyPatch,
    *,
    resolver: PathResolver,
    checkpoint_path: Path,
    output_dir: Path,
    test_calls: list[dict[str, object]],
) -> object:
    trainer_config = SimpleNamespace(
        precision="32-true",
        deterministic=True,
        benchmark=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    court_runtime = SimpleNamespace(
        runtime=SimpleNamespace(
            resolver=resolver,
            training=SimpleNamespace(trainer=trainer_config),
        ),
        evaluation_checkpoint=checkpoint_path,
    )

    class _RuntimeBoundary:
        @classmethod
        def from_config(
            cls,
            config: object,
            *,
            evaluation: bool = False,
        ) -> SimpleNamespace:
            del cls, config
            assert evaluation is True
            return court_runtime

    class _Trainer:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def test(self, *args: object, **kwargs: object) -> None:
            del args
            test_calls.append(kwargs)

    datamodule = object()
    module = object()
    monkeypatch.setattr(
        "src.tasks.court_alignment.training.runner.CourtAlignmentRuntimeConfig",
        _RuntimeBoundary,
    )
    monkeypatch.setattr(
        "src.tasks.court_alignment.training.runner.pl.Trainer",
        _Trainer,
    )
    monkeypatch.setattr(
        "src.tasks.court_alignment.training.runner.resolve_queue_repro_dir",
        lambda: None,
    )
    monkeypatch.setattr(runner, "seed_everything", lambda _: None)
    monkeypatch.setattr(runner, "apply_runtime_settings", lambda _: None)
    monkeypatch.setattr(runner, "prepare_output_dir", lambda _: output_dir)
    monkeypatch.setattr(runner, "save_config", lambda *_: None)
    monkeypatch.setattr(runner, "build_datamodule", lambda _: cast(Any, datamodule))
    monkeypatch.setattr(
        runner,
        "build_lightning_module",
        lambda *_, **__: cast(Any, module),
    )
    monkeypatch.setattr(runner, "select_devices", lambda _: ("cpu", 1))
    return datamodule


def test_test_checkpoint_path_selects_nonempty_validation_best(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CourtAlignmentTrainingRunner()
    monkeypatch.setattr(
        runner,
        "validate_runtime_config",
        lambda _: _runtime(checkpoint_enabled=True),
    )
    checkpoint = ModelCheckpoint()
    checkpoint.best_model_path = "/tmp/best.ckpt"
    trainer = SimpleNamespace(checkpoint_callback=checkpoint)

    assert runner.test_checkpoint_path({}, cast(Any, trainer)) == "best"


@pytest.mark.parametrize(
    "trainer",
    [
        SimpleNamespace(checkpoint_callback=None),
        SimpleNamespace(checkpoint_callback=ModelCheckpoint()),
    ],
)
def test_test_checkpoint_path_rejects_missing_validation_best(
    trainer: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CourtAlignmentTrainingRunner()
    monkeypatch.setattr(
        runner,
        "validate_runtime_config",
        lambda _: _runtime(checkpoint_enabled=True),
    )

    with pytest.raises(RuntimeError, match="[Cc]heckpoint"):
        runner.test_checkpoint_path({}, cast(Any, trainer))


def test_test_checkpoint_path_rejects_disabled_checkpointing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CourtAlignmentTrainingRunner()
    monkeypatch.setattr(
        runner,
        "validate_runtime_config",
        lambda _: _runtime(checkpoint_enabled=False),
    )

    with pytest.raises(RuntimeError, match="without checkpointing"):
        runner.test_checkpoint_path({}, cast(Any, SimpleNamespace()))


def test_evaluate_passes_contained_checkpoint_and_explicit_full_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver = _path_resolver(tmp_path)
    checkpoint_path = resolver.resolve(PathRole.CHECKPOINT, "model.ckpt")
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"checkpoint")
    runner = CourtAlignmentTrainingRunner()
    test_calls: list[dict[str, object]] = []
    datamodule = _install_evaluation_runtime(
        runner,
        monkeypatch,
        resolver=resolver,
        checkpoint_path=checkpoint_path,
        output_dir=tmp_path / "evaluation-output",
        test_calls=test_calls,
    )

    runner.evaluate(cast(Any, {}))

    assert len(test_calls) == 1
    assert test_calls[0]["datamodule"] is datamodule
    assert test_calls[0]["ckpt_path"] == str(checkpoint_path.resolve())
    assert test_calls[0]["weights_only"] is False


def test_evaluate_rejects_checkpoint_outside_role_before_trainer_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver = _path_resolver(tmp_path)
    outside_path = (tmp_path / "outside.ckpt").resolve()
    outside_path.write_bytes(b"checkpoint")
    runner = CourtAlignmentTrainingRunner()
    test_calls: list[dict[str, object]] = []
    _install_evaluation_runtime(
        runner,
        monkeypatch,
        resolver=resolver,
        checkpoint_path=outside_path,
        output_dir=tmp_path / "evaluation-output",
        test_calls=test_calls,
    )

    with pytest.raises(PathContractError, match="outside its root"):
        runner.evaluate(cast(Any, {}))

    assert test_calls == []
    assert not (tmp_path / "evaluation-output").exists()
