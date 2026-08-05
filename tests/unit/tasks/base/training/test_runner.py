"""Unit tests for config-driven BaseTrainingRunner construction."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.tasks.base.training.runner import BaseTrainingRunner
from src.utils.configuration import PathContractError


def test_build_trainer_forwards_dataloader_reload_interval(
    make_training_config: Any,
) -> None:
    config = OmegaConf.create(
        make_training_config(
            trainer={"reload_dataloaders_every_n_epochs": 1}
        )
    )

    trainer = BaseTrainingRunner().build_trainer(
        config,
        callbacks=[],
        logger=cast(Any, False),
    )

    assert trainer.reload_dataloaders_every_n_epochs == 1


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
