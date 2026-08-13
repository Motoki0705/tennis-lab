"""Unit tests for config-driven BaseTrainingRunner construction."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, cast

import pytest
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.tasks.base.training.repro import QueueReproDirError
from src.tasks.base.training.runner import BaseTrainingRunner
from src.utils.configuration import PathContractError
from src.utils.device import DeviceSelectionError


class _TrainerWithDataloaderReload(Protocol):
    reload_dataloaders_every_n_epochs: int


class _RecordingTrainer:
    def __init__(self) -> None:
        self.test_calls: list[tuple[object, dict[str, object]]] = []

    def test(self, model: object, **kwargs: object) -> None:
        self.test_calls.append((model, kwargs))


def _checkpoint_enabled_config(make_training_config: Any, **run: bool) -> Any:
    config = OmegaConf.create(make_training_config(run=run))
    config.training.checkpoint.enabled = True
    return config


def test_build_trainer_forwards_dataloader_reload_interval(
    make_training_config: Any,
) -> None:
    config = OmegaConf.create(
        make_training_config(trainer={"reload_dataloaders_every_n_epochs": 1})
    )

    trainer = BaseTrainingRunner().build_trainer(
        config,
        callbacks=[],
        logger=cast(Any, False),
    )

    inspected_trainer = cast(_TrainerWithDataloaderReload, trainer)
    assert inspected_trainer.reload_dataloaders_every_n_epochs == 1


def test_test_after_fit_uses_validation_winner_when_checkpointing_enabled(
    make_training_config: Any,
) -> None:
    config = _checkpoint_enabled_config(
        make_training_config,
        test_after_fit=True,
    )
    runner = BaseTrainingRunner()
    runtime = runner.validate_runtime_config(config)
    trainer = _RecordingTrainer()
    module = object()
    datamodule = object()

    runner.run_test_after_fit(
        trainer=cast(Any, trainer),
        lightning_module=cast(Any, module),
        datamodule=cast(Any, datamodule),
        config=config,
        runtime=runtime,
    )

    assert trainer.test_calls == [
        (
            module,
            {
                "datamodule": datamodule,
                "ckpt_path": "best",
                "weights_only": False,
            },
        )
    ]


def test_test_after_fit_uses_current_weights_when_checkpointing_disabled(
    make_training_config: Any,
) -> None:
    config = OmegaConf.create(make_training_config(run={"test_after_fit": True}))
    runner = BaseTrainingRunner()
    runtime = runner.validate_runtime_config(config)
    trainer = _RecordingTrainer()
    module = object()
    datamodule = object()

    runner.run_test_after_fit(
        trainer=cast(Any, trainer),
        lightning_module=cast(Any, module),
        datamodule=cast(Any, datamodule),
        config=config,
        runtime=runtime,
    )

    assert trainer.test_calls == [(module, {"datamodule": datamodule})]


@pytest.mark.parametrize(
    "run",
    [
        {"test_after_fit": False, "fast_dev_run": False},
        {"test_after_fit": True, "fast_dev_run": True},
    ],
)
def test_test_after_fit_preserves_explicit_skip_modes(
    make_training_config: Any,
    run: dict[str, bool],
) -> None:
    config = _checkpoint_enabled_config(make_training_config, **run)
    runner = BaseTrainingRunner()
    trainer = _RecordingTrainer()

    runner.run_test_after_fit(
        trainer=cast(Any, trainer),
        lightning_module=cast(Any, object()),
        datamodule=cast(Any, object()),
        config=config,
        runtime=runner.validate_runtime_config(config),
    )

    assert trainer.test_calls == []


def test_select_devices_rejects_unavailable_positive_gpu_request(
    make_training_config: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = OmegaConf.create(make_training_config(run={"gpus": 1}))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(DeviceSelectionError, match="explicitly requests GPU"):
        BaseTrainingRunner().select_devices(config)


def test_build_trainer_rejects_unavailable_gpu_before_trainer_construction(
    make_training_config: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = OmegaConf.create(make_training_config(run={"gpus": 1}))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    constructions: list[dict[str, object]] = []
    monkeypatch.setattr(
        "src.tasks.base.training.runner.pl.Trainer",
        lambda **kwargs: constructions.append(kwargs),
    )

    with pytest.raises(DeviceSelectionError, match="explicitly requests GPU"):
        BaseTrainingRunner().build_trainer(
            config,
            callbacks=[],
            logger=cast(Any, False),
        )

    assert constructions == []


def test_build_callbacks_forwards_explicit_early_stopping_timing(
    make_training_config: Any,
) -> None:
    config = OmegaConf.create(make_training_config())
    config.training.early_stopping.enabled = True
    config.training.early_stopping.min_delta = 0.25

    callbacks = BaseTrainingRunner().build_callbacks(
        config,
        datamodule=cast(Any, object()),
        logger=cast(Any, object()),
    )

    early_stopping = next(
        callback for callback in callbacks if isinstance(callback, EarlyStopping)
    )
    assert early_stopping._check_on_train_epoch_end is False
    assert early_stopping.min_delta == -0.25


def test_save_config_resolves_fixed_child_beneath_output_role(
    make_training_config: Any,
) -> None:
    config = OmegaConf.create(make_training_config())
    runner = BaseTrainingRunner()
    output_dir = runner.validate_runtime_config(config).run.output_dir
    output_dir.mkdir(parents=True)

    runner.save_config(config, output_dir)

    assert (output_dir / "config.yaml").is_file()


def test_save_config_rejects_output_parent_outside_role_root(
    make_training_config: Any,
    tmp_path: Path,
) -> None:
    config = OmegaConf.create(make_training_config())

    with pytest.raises(PathContractError, match="outside its root"):
        BaseTrainingRunner().save_config(config, tmp_path / "outside")


def test_build_logger_rejects_output_parent_outside_role_root(
    make_training_config: Any,
    tmp_path: Path,
) -> None:
    config = OmegaConf.create(make_training_config())

    with pytest.raises(PathContractError, match="outside its root"):
        BaseTrainingRunner().build_logger(config, tmp_path / "outside")


def test_checkpoint_dir_resolves_beneath_validated_logger_parent(
    make_training_config: Any,
) -> None:
    config = OmegaConf.create(make_training_config())
    config.training.checkpoint.enabled = True
    runner = BaseTrainingRunner()
    output_dir = runner.validate_runtime_config(config).run.output_dir
    log_dir = output_dir / "logs" / "version_0"
    logger = cast(Any, SimpleNamespace(log_dir=str(log_dir)))

    callbacks = runner.build_callbacks(
        config,
        datamodule=cast(Any, object()),
        logger=logger,
    )

    checkpoint = next(
        callback for callback in callbacks if isinstance(callback, ModelCheckpoint)
    )
    assert checkpoint.dirpath == str(log_dir / "checkpoints")


def test_checkpoint_pointer_preserves_legacy_artifact_location_without_queue(
    make_training_config: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TENNIS_REPRO_DIR", raising=False)
    runner = BaseTrainingRunner()
    runtime = runner.validate_runtime_config(OmegaConf.create(make_training_config()))
    checkpoint_dir = runtime.resolver.roots.output_root / "run" / "checkpoints"

    runner._record_ckpt_dir_pointer(checkpoint_dir, runtime.resolver)

    pointer = runtime.resolver.roots.artifact_root / "repro" / "output_dir.txt"
    assert pointer.read_text(encoding="utf-8") == f"{checkpoint_dir.resolve()}\n"


def test_checkpoint_pointer_isolated_between_queue_repro_dirs(
    make_training_config: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = BaseTrainingRunner()
    runtime = runner.validate_runtime_config(OmegaConf.create(make_training_config()))
    repro_dirs = (tmp_path / "queue-a", tmp_path / "queue-b")
    checkpoint_dirs = (tmp_path / "checkpoint-a", tmp_path / "checkpoint-b")

    for repro_dir, checkpoint_dir in zip(
        repro_dirs, checkpoint_dirs, strict=True
    ):
        monkeypatch.setenv("TENNIS_REPRO_DIR", str(repro_dir))
        runner._record_ckpt_dir_pointer(checkpoint_dir, runtime.resolver)

    for repro_dir, checkpoint_dir in zip(
        repro_dirs, checkpoint_dirs, strict=True
    ):
        assert (repro_dir / "output_dir.txt").read_text(
            encoding="utf-8"
        ) == f"{checkpoint_dir.resolve()}\n"
    assert not (runtime.resolver.roots.artifact_root / "repro").exists()


def test_checkpoint_pointer_rejects_invalid_queue_dir_before_writes(
    make_training_config: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = BaseTrainingRunner()
    runtime = runner.validate_runtime_config(OmegaConf.create(make_training_config()))
    monkeypatch.setenv("TENNIS_REPRO_DIR", "relative/repro")

    with pytest.raises(QueueReproDirError, match="absolute"):
        runner._record_ckpt_dir_pointer(tmp_path / "checkpoints", runtime.resolver)

    assert not (runtime.resolver.roots.artifact_root / "repro").exists()


def test_checkpoint_dir_rejects_logger_parent_outside_output_role(
    make_training_config: Any,
    tmp_path: Path,
) -> None:
    config = OmegaConf.create(make_training_config())
    config.training.checkpoint.enabled = True

    with pytest.raises(PathContractError, match="outside its root"):
        BaseTrainingRunner().build_callbacks(
            config,
            datamodule=cast(Any, object()),
            logger=cast(Any, SimpleNamespace(log_dir=str(tmp_path / "outside"))),
        )
