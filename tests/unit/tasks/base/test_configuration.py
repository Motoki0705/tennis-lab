"""Unit tests for shared strict training configuration semantics."""

from __future__ import annotations

from typing import Any, cast

import pytest

from src.tasks.base.configuration import TrainingRuntimeConfig
from src.utils.configuration import (
    ConfigurationTypeError,
    SemanticConfigurationError,
)

pytestmark = pytest.mark.unit


def _set(config: dict[str, Any], dotted_path: str, value: object) -> None:
    parts = dotted_path.split(".")
    current = config
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value


def _validate(config: dict[str, Any], tmp_path: Any) -> TrainingRuntimeConfig:
    return TrainingRuntimeConfig.from_config(config, repository_root=tmp_path)


def test_complete_shared_training_contract_is_accepted(
    make_training_config: Any,
    tmp_path: Any,
) -> None:
    runtime = _validate(make_training_config(), tmp_path)

    assert runtime.training.trainer.precision == "32-true"
    assert runtime.training.optimizer.betas == (0.9, 0.999)
    assert runtime.training.early_stopping.check_on_train_epoch_end is False


@pytest.mark.parametrize(
    ("dotted_path", "value"),
    [
        ("run.seed", -1),
        ("run.seed", 2**32),
        ("training.trainer.gradient_clip_val", -0.1),
        ("training.trainer.gradient_clip_val", float("inf")),
        ("training.trainer.deterministic", "strict"),
        ("training.trainer.precision", "16"),
        ("training.learning_rate", float("inf")),
        ("training.weight_decay", float("nan")),
        ("training.min_lr", 2.0e-3),
        ("training.optimizer.betas", [1.0, 0.999]),
        ("training.optimizer.betas", [0.9, float("nan")]),
        ("training.checkpoint.filename", "../model-{epoch}"),
        ("training.checkpoint.monitor", " "),
        ("training.checkpoint.save_top_k", -2),
        ("training.early_stopping.monitor", ""),
        ("training.early_stopping.min_delta", -0.1),
        ("training.gan.generator_gradient_clip_val", -1.0),
        ("training.gan.discriminator_gradient_clip_val", float("inf")),
        ("training.matmul_precision", "default"),
    ],
)
def test_invalid_shared_training_semantics_fail_at_boundary(
    make_training_config: Any,
    tmp_path: Any,
    dotted_path: str,
    value: object,
) -> None:
    config = make_training_config()
    _set(config, dotted_path, value)

    with pytest.raises(SemanticConfigurationError, match=dotted_path.rsplit(".", 1)[0]):
        _validate(config, tmp_path)


@pytest.mark.parametrize(
    ("dotted_path", "value"),
    [
        ("training.trainer.precision", None),
        ("training.trainer.precision", 32),
        ("training.optimizer.betas", None),
        ("training.early_stopping.check_on_train_epoch_end", None),
    ],
)
def test_framework_default_delegation_is_rejected(
    make_training_config: Any,
    tmp_path: Any,
    dotted_path: str,
    value: object,
) -> None:
    config = make_training_config()
    _set(config, dotted_path, value)

    with pytest.raises(ConfigurationTypeError, match=dotted_path):
        _validate(config, tmp_path)


def test_run_modes_are_mutually_exclusive(
    make_training_config: Any,
    tmp_path: Any,
) -> None:
    config = make_training_config(
        run={"fast_dev_run": True, "dry_run": True},
    )

    with pytest.raises(SemanticConfigurationError, match="mutually exclusive"):
        _validate(config, tmp_path)


def test_deterministic_execution_rejects_benchmark_mode(
    make_training_config: Any,
    tmp_path: Any,
) -> None:
    config = make_training_config(
        trainer={"deterministic": "warn", "benchmark": True},
    )

    with pytest.raises(SemanticConfigurationError, match="benchmark"):
        _validate(config, tmp_path)


def test_enabled_checkpoint_must_write_an_artifact(
    make_training_config: Any,
    tmp_path: Any,
) -> None:
    config = make_training_config()
    config["training"]["checkpoint"].update(
        {"enabled": True, "save_top_k": 0, "save_last": False}
    )

    with pytest.raises(SemanticConfigurationError, match="save at least"):
        _validate(config, tmp_path)


def _enabled_gan_config(make_training_config: Any) -> dict[str, Any]:
    config = cast(
        "dict[str, Any]", make_training_config(trainer={"max_epochs": 4})
    )
    config["training"]["trainer"]["gradient_clip_val"] = None
    config["training"]["early_stopping"]["enabled"] = False
    config["training"]["gan"].update(
        {
            "enabled": True,
            "target_weight": 0.1,
            "warmup_epochs": 2,
            "transition": {"start_epoch": 1},
        }
    )
    return config


def test_explicit_gan_contract_is_accepted(
    make_training_config: Any,
    tmp_path: Any,
) -> None:
    runtime = _validate(_enabled_gan_config(make_training_config), tmp_path)

    assert runtime.training.gan.enabled is True


@pytest.mark.parametrize(
    ("dotted_path", "value", "message"),
    [
        ("training.gan.target_weight", 0.0, "target_weight"),
        ("training.gan.transition.start_epoch", 4, "start_epoch"),
        ("training.gan.warmup_epochs", 4, "finish"),
        ("training.trainer.gradient_clip_val", 1.0, "gradient_clip_val"),
        ("training.early_stopping.enabled", True, "early_stopping"),
    ],
)
def test_enabled_gan_rejects_conflicting_settings(
    make_training_config: Any,
    tmp_path: Any,
    dotted_path: str,
    value: object,
    message: str,
) -> None:
    config = _enabled_gan_config(make_training_config)
    _set(config, dotted_path, value)

    with pytest.raises(SemanticConfigurationError, match=message):
        _validate(config, tmp_path)


def test_fixed_qualitative_indices_are_explicit_and_consistent(
    make_training_config: Any,
    tmp_path: Any,
) -> None:
    config = make_training_config()
    config["training"]["qualitative_logging"].update(
        {
            "selection_mode": "fixed_indices",
            "selected_indices": [0, 2],
            "num_samples": 2,
        }
    )

    runtime = _validate(config, tmp_path)
    assert runtime.training.qualitative_logging.selected_indices == (0, 2)
