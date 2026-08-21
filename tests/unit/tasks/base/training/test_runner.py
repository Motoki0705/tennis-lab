"""Unit tests for config-driven BaseTrainingRunner construction."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, cast

import pytest
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.tasks.base.training.runner import BaseTrainingRunner
from src.utils.configuration import PathContractError
from src.utils.device import DeviceSelectionError


class _TrainerWithDataloaderReload(Protocol):
    reload_dataloaders_every_n_epochs: int


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


def test_run_compiles_after_init_weights_and_before_trainer_construction(
    make_training_config: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = OmegaConf.create(make_training_config())
    runner = BaseTrainingRunner()
    runtime = runner.validate_runtime_config(config)
    events: list[str] = []
    lightning_module = cast(Any, object())
    datamodule = cast(Any, object())

    class _Trainer:
        def fit(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            events.append("fit")

    monkeypatch.setattr(runner, "prepare_config", lambda _: None)
    monkeypatch.setattr(runner, "seed_everything", lambda _: None)
    monkeypatch.setattr(runner, "apply_runtime_settings", lambda _: None)
    monkeypatch.setattr(runner, "save_config", lambda *_: None)
    monkeypatch.setattr(runner, "build_datamodule", lambda _: datamodule)
    monkeypatch.setattr(runner, "resolve_steps_per_epoch", lambda *_, **__: None)
    monkeypatch.setattr(
        runner, "build_lightning_module", lambda *_, **__: lightning_module
    )
    monkeypatch.setattr(
        runner,
        "maybe_load_init_weights",
        lambda *_: events.append("init_weights"),
    )
    def compile_models(*args: object) -> tuple[str, ...]:
        del args
        events.append("compile")
        return ("model",)

    monkeypatch.setattr(runner, "maybe_compile_models", compile_models)
    monkeypatch.setattr(runner, "build_logger", lambda *_: cast(Any, False))
    monkeypatch.setattr(runner, "build_callbacks", lambda *_: [])

    def build_trainer(*args: object, **kwargs: object) -> _Trainer:
        del args, kwargs
        events.append("trainer")
        return _Trainer()

    monkeypatch.setattr(runner, "build_trainer", build_trainer)
    monkeypatch.setattr(runner, "resolve_resume", lambda *_: None)

    runner.run(config)

    assert runtime.training.compile.enabled is True
    assert events == ["init_weights", "compile", "trainer", "fit"]


def test_maybe_compile_models_is_noop_when_disabled(
    make_training_config: Any,
) -> None:
    config = OmegaConf.create(
        make_training_config(
            training={
                "compile": {
                    "enabled": False,
                    "backend": "inductor",
                    "mode": "reduce-overhead",
                    "fullgraph": False,
                    "dynamic": False,
                }
            }
        )
    )
    runtime = BaseTrainingRunner().validate_runtime_config(config)

    assert (
        BaseTrainingRunner().maybe_compile_models(
            runtime,
            cast(Any, object()),
        )
        == ()
    )


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
