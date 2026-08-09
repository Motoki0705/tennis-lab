"""Strict task-runtime contracts shared by training, data, and visualization.

The contracts in this module own only settings whose meaning is shared by
multiple tasks.  Task-specific model, loss, augmentation, and dataset fields
remain in their task packages.  Callers must provide every shared field; this
module never supplies a value which belongs in composed configuration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

from omegaconf import DictConfig, OmegaConf

from src.utils.configuration import (
    ConfigurationTypeError,
    MissingConfigurationKeyError,
    PathResolver,
    PathRole,
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)

ConfigMapping: TypeAlias = Mapping[str, object]
TrainerDeterministic: TypeAlias = bool | Literal["warn"]
TrainerPrecision: TypeAlias = Literal[
    "transformer-engine",
    "transformer-engine-float16",
    "16-true",
    "16-mixed",
    "bf16-true",
    "bf16-mixed",
    "32-true",
    "64-true",
]
MonitorMode: TypeAlias = Literal["min", "max"]

__all__ = [
    "BaseDataConfig",
    "BaseRunConfig",
    "BaseTrainingConfig",
    "CheckpointConfig",
    "ChunkDataConfig",
    "EarlyStoppingConfig",
    "GANConfig",
    "GANTransitionConfig",
    "LRMonitorConfig",
    "OptimizerConfig",
    "QualitativeLoggingConfig",
    "SceneVisualizationConfig",
    "TrainerConfig",
    "TrainingRuntimeConfig",
    "as_config_mapping",
    "exact_config_mapping",
    "require_config_mapping",
    "require_config_value",
]


_RUN_KEYS = frozenset(
    {
        "output_dir",
        "seed",
        "gpus",
        "resume",
        "init_weights",
        "fast_dev_run",
        "dry_run",
        "test_after_fit",
    }
)
_TRAINER_KEYS = frozenset(
    {
        "max_epochs",
        "gradient_clip_val",
        "deterministic",
        "precision",
        "log_every_n_steps",
        "check_val_every_n_epoch",
        "accumulate_grad_batches",
        "reload_dataloaders_every_n_epochs",
        "enable_progress_bar",
        "enable_model_summary",
        "benchmark",
    }
)
_BASE_TRAINING_KEYS = frozenset(
    {
        "trainer",
        "learning_rate",
        "weight_decay",
        "warmup_steps",
        "warmup_epochs",
        "min_lr",
        "steps_per_epoch",
        "optimizer",
        "checkpoint",
        "early_stopping",
        "lr_monitor",
        "qualitative_logging",
        "gan",
        "matmul_precision",
        "allow_tf32",
    }
)
_OPTIMIZER_KEYS = frozenset({"betas"})
_CHECKPOINT_KEYS = frozenset(
    {"enabled", "filename", "monitor", "mode", "save_top_k", "save_last"}
)
_EARLY_STOPPING_KEYS = frozenset(
    {"enabled", "monitor", "mode", "patience", "min_delta", "check_on_train_epoch_end"}
)
_LR_MONITOR_KEYS = frozenset({"enabled", "interval"})
_QUALITATIVE_LOGGING_KEYS = frozenset(
    {"enabled", "every_n_epochs", "num_samples", "selection_mode", "selected_indices"}
)
_GAN_KEYS = frozenset(
    {
        "enabled",
        "target_weight",
        "warmup_epochs",
        "generator_gradient_clip_val",
        "discriminator_gradient_clip_val",
        "transition",
    }
)
_GAN_TRANSITION_KEYS = frozenset({"start_epoch"})
_BASE_DATA_KEYS = frozenset({"scene_dir", "batch_size", "num_workers", "pin_memory"})
_CHUNK_DATA_KEYS = frozenset({"chunk", "generator_device"})
_CHUNK_KEYS = frozenset(
    {
        "scenes_per_chunk",
        "epochs_per_chunk",
        "prefetch_chunks",
        "chunks_dir",
        "generation_workers",
    }
)
_TRAINER_PRECISIONS = frozenset(
    {
        "transformer-engine",
        "transformer-engine-float16",
        "16-true",
        "16-mixed",
        "bf16-true",
        "bf16-mixed",
        "32-true",
        "64-true",
    }
)
_MATMUL_PRECISIONS = frozenset({"highest", "high", "medium"})
_MAX_LIGHTNING_SEED = 2**32 - 1


def as_config_mapping(value: object, *, path: str) -> ConfigMapping:
    """Return a resolved plain mapping or raise a configuration type error."""
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping):
        raise ConfigurationTypeError(
            f"{path}: expected mapping, got {type(value).__name__}."
        )
    non_string_keys = tuple(key for key in value if not isinstance(key, str))
    if non_string_keys:
        raise ConfigurationTypeError(
            f"{path}: all keys must be strings; got {non_string_keys!r}."
        )
    return cast("ConfigMapping", value)


def exact_config_mapping(
    value: object,
    *,
    path: str,
    required_keys: Set[str],
    optional_keys: Set[str] = frozenset(),
) -> ConfigMapping:
    """Return an exact-key mapping, rejecting unknown keys before any reads."""
    mapping = as_config_mapping(value, path=path)
    unknown = sorted(set(mapping) - set(required_keys) - set(optional_keys))
    if unknown:
        rendered = ", ".join(f"{path}.{key}" for key in unknown)
        raise UnknownConfigurationKeyError(f"Unknown configuration key(s): {rendered}.")
    missing = sorted(set(required_keys) - set(mapping))
    if missing:
        rendered = ", ".join(f"{path}.{key}" for key in missing)
        raise MissingConfigurationKeyError(
            f"Missing required configuration key(s): {rendered}."
        )
    return mapping


def _project_required_mapping(
    value: object,
    *,
    path: str,
    keys: Set[str],
) -> ConfigMapping:
    """Project fields from a task mapping whose full schema is task-owned.

    Domain boundaries must exact-validate the complete mapping.  This helper
    only constructs the closed shared sub-contract consumed by this module.
    """
    mapping = as_config_mapping(value, path=path)
    missing = sorted(set(keys) - set(mapping))
    if missing:
        rendered = ", ".join(f"{path}.{key}" for key in missing)
        raise MissingConfigurationKeyError(
            f"Missing required configuration key(s): {rendered}."
        )
    return {key: mapping[key] for key in keys}


def require_config_value(
    mapping: ConfigMapping,
    key: str,
    expected_types: type[object] | tuple[type[object], ...],
    *,
    path: str,
) -> object:
    """Read one required exact-typed value without synthesizing a default."""
    try:
        value = mapping[key]
    except KeyError as error:
        raise MissingConfigurationKeyError(
            f"Missing required configuration key: {path}.{key}."
        ) from error
    accepted = (
        expected_types if isinstance(expected_types, tuple) else (expected_types,)
    )
    if type(value) not in accepted:
        names = " | ".join(candidate.__name__ for candidate in accepted)
        raise ConfigurationTypeError(
            f"{path}.{key}: expected {names}, got {type(value).__name__}."
        )
    return value


def require_config_mapping(
    mapping: ConfigMapping,
    key: str,
    *,
    path: str,
) -> ConfigMapping:
    """Read one required mapping child without a missing-section fallback."""
    return as_config_mapping(
        require_config_value(mapping, key, (dict, DictConfig), path=path),
        path=f"{path}.{key}",
    )


def _optional_path(
    mapping: ConfigMapping,
    key: str,
    *,
    path: str,
    resolver: PathResolver,
    role: PathRole,
) -> Path | None:
    raw = require_config_value(mapping, key, (str, type(None)), path=path)
    if raw is None:
        return None
    if raw == "":
        raise SemanticConfigurationError(f"{path}.{key}: path must not be empty.")
    resolved: Path = resolver.resolve(role, cast("str", raw))
    return resolved


def _positive(value: int | float, *, path: str, allow_zero: bool = False) -> None:
    if not math.isfinite(float(value)):
        raise SemanticConfigurationError(f"{path} must be finite; got {value!r}.")
    valid = value >= 0 if allow_zero else value > 0
    if not valid:
        operator = ">=" if allow_zero else ">"
        raise SemanticConfigurationError(f"{path} must be {operator} 0; got {value!r}.")


def _non_empty(value: str, *, path: str) -> None:
    if not value or value != value.strip():
        raise SemanticConfigurationError(
            f"{path} must be a non-empty, trimmed string; got {value!r}."
        )


def _optional_non_negative(
    value: int | float | None,
    *,
    path: str,
) -> None:
    if value is not None:
        _positive(value, path=path, allow_zero=True)


@dataclass(frozen=True, slots=True)
class BaseRunConfig:
    """Shared training-run settings, with role-based resolved paths."""

    output_dir: Path
    seed: int
    gpus: int
    resume: Path | None
    init_weights: Path | None
    fast_dev_run: bool
    dry_run: bool
    test_after_fit: bool

    @classmethod
    def from_mapping(cls, value: object, *, resolver: PathResolver) -> BaseRunConfig:
        """Validate a composed ``run`` section before training starts."""
        mapping = exact_config_mapping(value, path="run", required_keys=_RUN_KEYS)
        output = cast(
            "str", require_config_value(mapping, "output_dir", str, path="run")
        )
        if not output:
            raise SemanticConfigurationError("run.output_dir must not be empty.")
        resume = _optional_path(
            mapping, "resume", path="run", resolver=resolver, role=PathRole.CHECKPOINT
        )
        init_weights = _optional_path(
            mapping,
            "init_weights",
            path="run",
            resolver=resolver,
            role=PathRole.CHECKPOINT,
        )
        if resume is not None and init_weights is not None:
            raise SemanticConfigurationError(
                "run.resume and run.init_weights are mutually exclusive."
            )
        gpus = cast("int", require_config_value(mapping, "gpus", int, path="run"))
        _positive(gpus, path="run.gpus", allow_zero=True)
        seed = cast("int", require_config_value(mapping, "seed", int, path="run"))
        if not 0 <= seed <= _MAX_LIGHTNING_SEED:
            raise SemanticConfigurationError(
                f"run.seed must be between 0 and {_MAX_LIGHTNING_SEED}; got {seed!r}."
            )
        fast_dev_run = cast(
            "bool", require_config_value(mapping, "fast_dev_run", bool, path="run")
        )
        dry_run = cast(
            "bool", require_config_value(mapping, "dry_run", bool, path="run")
        )
        if fast_dev_run and dry_run:
            raise SemanticConfigurationError(
                "run.fast_dev_run and run.dry_run are mutually exclusive."
            )
        return cls(
            output_dir=resolver.resolve(PathRole.OUTPUT, output),
            seed=seed,
            gpus=gpus,
            resume=resume,
            init_weights=init_weights,
            fast_dev_run=fast_dev_run,
            dry_run=dry_run,
            test_after_fit=cast(
                "bool",
                require_config_value(mapping, "test_after_fit", bool, path="run"),
            ),
        )


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Lightning Trainer fields consumed by the shared runner."""

    max_epochs: int
    gradient_clip_val: float | int | None
    deterministic: TrainerDeterministic
    precision: TrainerPrecision
    log_every_n_steps: int
    check_val_every_n_epoch: int
    accumulate_grad_batches: int
    reload_dataloaders_every_n_epochs: int
    enable_progress_bar: bool
    enable_model_summary: bool
    benchmark: bool

    @classmethod
    def from_mapping(cls, value: object) -> TrainerConfig:
        mapping = exact_config_mapping(
            value,
            path="training.trainer",
            required_keys=_TRAINER_KEYS,
        )
        deterministic = require_config_value(
            mapping, "deterministic", (bool, str), path="training.trainer"
        )
        if isinstance(deterministic, str) and deterministic != "warn":
            raise SemanticConfigurationError(
                "training.trainer.deterministic must be true, false, or 'warn'."
            )
        precision = cast(
            "str",
            require_config_value(
                mapping,
                "precision",
                str,
                path="training.trainer",
            ),
        )
        if precision not in _TRAINER_PRECISIONS:
            allowed = ", ".join(sorted(_TRAINER_PRECISIONS))
            raise SemanticConfigurationError(
                "training.trainer.precision must use a canonical Lightning value; "
                f"got {precision!r}. Allowed: {allowed}."
            )
        result = cls(
            max_epochs=cast(
                "int",
                require_config_value(
                    mapping, "max_epochs", int, path="training.trainer"
                ),
            ),
            gradient_clip_val=cast(
                "float | int | None",
                require_config_value(
                    mapping,
                    "gradient_clip_val",
                    (float, int, type(None)),
                    path="training.trainer",
                ),
            ),
            deterministic=cast("TrainerDeterministic", deterministic),
            precision=cast("TrainerPrecision", precision),
            log_every_n_steps=cast(
                "int",
                require_config_value(
                    mapping, "log_every_n_steps", int, path="training.trainer"
                ),
            ),
            check_val_every_n_epoch=cast(
                "int",
                require_config_value(
                    mapping,
                    "check_val_every_n_epoch",
                    int,
                    path="training.trainer",
                ),
            ),
            accumulate_grad_batches=cast(
                "int",
                require_config_value(
                    mapping,
                    "accumulate_grad_batches",
                    int,
                    path="training.trainer",
                ),
            ),
            reload_dataloaders_every_n_epochs=cast(
                "int",
                require_config_value(
                    mapping,
                    "reload_dataloaders_every_n_epochs",
                    int,
                    path="training.trainer",
                ),
            ),
            enable_progress_bar=cast(
                "bool",
                require_config_value(
                    mapping, "enable_progress_bar", bool, path="training.trainer"
                ),
            ),
            enable_model_summary=cast(
                "bool",
                require_config_value(
                    mapping, "enable_model_summary", bool, path="training.trainer"
                ),
            ),
            benchmark=cast(
                "bool",
                require_config_value(
                    mapping, "benchmark", bool, path="training.trainer"
                ),
            ),
        )
        _positive(result.max_epochs, path="training.trainer.max_epochs")
        _positive(result.log_every_n_steps, path="training.trainer.log_every_n_steps")
        _positive(
            result.check_val_every_n_epoch,
            path="training.trainer.check_val_every_n_epoch",
        )
        _positive(
            result.accumulate_grad_batches,
            path="training.trainer.accumulate_grad_batches",
        )
        _positive(
            result.reload_dataloaders_every_n_epochs,
            path="training.trainer.reload_dataloaders_every_n_epochs",
            allow_zero=True,
        )
        _optional_non_negative(
            result.gradient_clip_val,
            path="training.trainer.gradient_clip_val",
        )
        if result.deterministic is not False and result.benchmark:
            raise SemanticConfigurationError(
                "training.trainer.benchmark must be false when deterministic "
                "execution is enabled."
            )
        return result


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    """Optimizer and schedule fields consumed by the shared Lightning module."""

    learning_rate: float
    weight_decay: float
    warmup_steps: int | None
    warmup_epochs: int | None
    max_epochs: int
    min_lr: float
    betas: tuple[float, float]
    steps_per_epoch: int | None

    @classmethod
    def from_mapping(cls, value: object, *, max_epochs: int) -> OptimizerConfig:
        mapping = exact_config_mapping(
            value,
            path="training",
            required_keys=_BASE_TRAINING_KEYS,
        )
        optimizer = exact_config_mapping(
            require_config_mapping(mapping, "optimizer", path="training"),
            path="training.optimizer",
            required_keys=_OPTIMIZER_KEYS,
        )
        raw_betas = require_config_value(
            optimizer, "betas", (list, tuple), path="training.optimizer"
        )
        sequence = cast("Sequence[object]", raw_betas)
        if len(sequence) != 2 or any(
            type(item) not in (float, int) for item in sequence
        ):
            raise ConfigurationTypeError(
                "training.optimizer.betas must contain exactly two numbers."
            )
        betas = (
            float(cast("float | int", sequence[0])),
            float(cast("float | int", sequence[1])),
        )
        result = cls(
            learning_rate=float(
                cast(
                    "float | int",
                    require_config_value(
                        mapping, "learning_rate", (float, int), path="training"
                    ),
                )
            ),
            weight_decay=float(
                cast(
                    "float | int",
                    require_config_value(
                        mapping, "weight_decay", (float, int), path="training"
                    ),
                )
            ),
            warmup_steps=cast(
                "int | None",
                require_config_value(
                    mapping, "warmup_steps", (int, type(None)), path="training"
                ),
            ),
            warmup_epochs=cast(
                "int | None",
                require_config_value(
                    mapping, "warmup_epochs", (int, type(None)), path="training"
                ),
            ),
            max_epochs=max_epochs,
            min_lr=float(
                cast(
                    "float | int",
                    require_config_value(
                        mapping, "min_lr", (float, int), path="training"
                    ),
                )
            ),
            betas=betas,
            steps_per_epoch=cast(
                "int | None",
                require_config_value(
                    mapping, "steps_per_epoch", (int, type(None)), path="training"
                ),
            ),
        )
        _positive(result.learning_rate, path="training.learning_rate")
        _positive(result.weight_decay, path="training.weight_decay", allow_zero=True)
        _positive(result.max_epochs, path="training.trainer.max_epochs")
        _positive(result.min_lr, path="training.min_lr", allow_zero=True)
        if result.min_lr > result.learning_rate:
            raise SemanticConfigurationError(
                "training.min_lr must be <= training.learning_rate."
            )
        for index, beta in enumerate(result.betas):
            if not math.isfinite(beta) or not 0.0 <= beta < 1.0:
                raise SemanticConfigurationError(
                    "training.optimizer.betas values must be finite and in [0, 1); "
                    f"index {index} is {beta!r}."
                )
        if result.warmup_steps is not None:
            _positive(
                result.warmup_steps, path="training.warmup_steps", allow_zero=True
            )
        if result.warmup_epochs is not None:
            _positive(
                result.warmup_epochs, path="training.warmup_epochs", allow_zero=True
            )
        if result.steps_per_epoch is not None:
            _positive(result.steps_per_epoch, path="training.steps_per_epoch")
        if (result.warmup_steps is None) == (result.warmup_epochs is None):
            raise SemanticConfigurationError(
                "Exactly one of training.warmup_steps and training.warmup_epochs "
                "must be an explicit integer."
            )
        if (
            result.warmup_epochs is not None
            and result.warmup_epochs >= result.max_epochs
            and result.warmup_epochs != 0
        ):
            raise SemanticConfigurationError(
                "training.warmup_epochs must be 0 or less than "
                "training.trainer.max_epochs."
            )
        if (
            result.warmup_steps is not None
            and result.steps_per_epoch is not None
            and result.warmup_steps >= result.steps_per_epoch * result.max_epochs
            and result.warmup_steps != 0
        ):
            raise SemanticConfigurationError(
                "training.warmup_steps must be 0 or less than the configured "
                "total training steps."
            )
        return result


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    enabled: bool
    filename: str
    monitor: str
    mode: MonitorMode
    save_top_k: int
    save_last: bool

    @classmethod
    def from_mapping(cls, value: object) -> CheckpointConfig:
        mapping = exact_config_mapping(
            value,
            path="training.checkpoint",
            required_keys=_CHECKPOINT_KEYS,
        )
        result = cls(
            enabled=cast(
                "bool",
                require_config_value(
                    mapping, "enabled", bool, path="training.checkpoint"
                ),
            ),
            filename=cast(
                "str",
                require_config_value(
                    mapping, "filename", str, path="training.checkpoint"
                ),
            ),
            monitor=cast(
                "str",
                require_config_value(
                    mapping, "monitor", str, path="training.checkpoint"
                ),
            ),
            mode=cast(
                "MonitorMode",
                require_config_value(mapping, "mode", str, path="training.checkpoint"),
            ),
            save_top_k=cast(
                "int",
                require_config_value(
                    mapping, "save_top_k", int, path="training.checkpoint"
                ),
            ),
            save_last=cast(
                "bool",
                require_config_value(
                    mapping, "save_last", bool, path="training.checkpoint"
                ),
            ),
        )
        if result.mode not in {"min", "max"}:
            raise SemanticConfigurationError(
                "training.checkpoint.mode must be 'min' or 'max'."
            )
        _non_empty(result.filename, path="training.checkpoint.filename")
        if (
            Path(result.filename).is_absolute()
            or "/" in result.filename
            or "\\" in result.filename
            or result.filename in {".", ".."}
        ):
            raise SemanticConfigurationError(
                "training.checkpoint.filename must be a filename template, not a path."
            )
        _non_empty(result.monitor, path="training.checkpoint.monitor")
        if result.save_top_k < -1:
            raise SemanticConfigurationError(
                "training.checkpoint.save_top_k must be >= -1."
            )
        if result.enabled and result.save_top_k == 0 and not result.save_last:
            raise SemanticConfigurationError(
                "Enabled checkpointing must save at least one top-k or last checkpoint."
            )
        return result


@dataclass(frozen=True, slots=True)
class EarlyStoppingConfig:
    enabled: bool
    monitor: str
    mode: MonitorMode
    patience: int
    min_delta: float
    check_on_train_epoch_end: bool

    @classmethod
    def from_mapping(cls, value: object) -> EarlyStoppingConfig:
        mapping = exact_config_mapping(
            value,
            path="training.early_stopping",
            required_keys=_EARLY_STOPPING_KEYS,
        )
        result = cls(
            enabled=cast(
                "bool",
                require_config_value(
                    mapping, "enabled", bool, path="training.early_stopping"
                ),
            ),
            monitor=cast(
                "str",
                require_config_value(
                    mapping, "monitor", str, path="training.early_stopping"
                ),
            ),
            mode=cast(
                "MonitorMode",
                require_config_value(
                    mapping, "mode", str, path="training.early_stopping"
                ),
            ),
            patience=cast(
                "int",
                require_config_value(
                    mapping, "patience", int, path="training.early_stopping"
                ),
            ),
            min_delta=float(
                cast(
                    "float | int",
                    require_config_value(
                        mapping,
                        "min_delta",
                        (float, int),
                        path="training.early_stopping",
                    ),
                )
            ),
            check_on_train_epoch_end=cast(
                "bool",
                require_config_value(
                    mapping,
                    "check_on_train_epoch_end",
                    bool,
                    path="training.early_stopping",
                ),
            ),
        )
        if result.mode not in {"min", "max"}:
            raise SemanticConfigurationError(
                "training.early_stopping.mode must be 'min' or 'max'."
            )
        _positive(
            result.patience, path="training.early_stopping.patience", allow_zero=True
        )
        _positive(
            result.min_delta,
            path="training.early_stopping.min_delta",
            allow_zero=True,
        )
        _non_empty(result.monitor, path="training.early_stopping.monitor")
        return result


@dataclass(frozen=True, slots=True)
class LRMonitorConfig:
    enabled: bool
    interval: Literal["step", "epoch"]

    @classmethod
    def from_mapping(cls, value: object) -> LRMonitorConfig:
        mapping = exact_config_mapping(
            value,
            path="training.lr_monitor",
            required_keys=_LR_MONITOR_KEYS,
        )
        interval = cast(
            "str",
            require_config_value(mapping, "interval", str, path="training.lr_monitor"),
        )
        if interval not in {"step", "epoch"}:
            raise SemanticConfigurationError(
                "training.lr_monitor.interval must be 'step' or 'epoch'."
            )
        return cls(
            enabled=cast(
                "bool",
                require_config_value(
                    mapping, "enabled", bool, path="training.lr_monitor"
                ),
            ),
            interval=cast("Literal['step', 'epoch']", interval),
        )


@dataclass(frozen=True, slots=True)
class QualitativeLoggingConfig:
    enabled: bool
    every_n_epochs: int
    num_samples: int
    selection_mode: str
    selected_indices: tuple[int, ...] | None

    @classmethod
    def from_mapping(cls, value: object) -> QualitativeLoggingConfig:
        mapping = exact_config_mapping(
            value,
            path="training.qualitative_logging",
            required_keys=_QUALITATIVE_LOGGING_KEYS,
        )
        raw_indices = require_config_value(
            mapping,
            "selected_indices",
            (list, tuple, type(None)),
            path="training.qualitative_logging",
        )
        indices: tuple[int, ...] | None = None
        if raw_indices is not None:
            sequence = cast("Sequence[object]", raw_indices)
            if any(type(item) is not int for item in sequence):
                raise ConfigurationTypeError(
                    "training.qualitative_logging.selected_indices must contain integers."
                )
            indices = tuple(cast("int", item) for item in sequence)
        result = cls(
            enabled=cast(
                "bool",
                require_config_value(
                    mapping, "enabled", bool, path="training.qualitative_logging"
                ),
            ),
            every_n_epochs=cast(
                "int",
                require_config_value(
                    mapping, "every_n_epochs", int, path="training.qualitative_logging"
                ),
            ),
            num_samples=cast(
                "int",
                require_config_value(
                    mapping, "num_samples", int, path="training.qualitative_logging"
                ),
            ),
            selection_mode=cast(
                "str",
                require_config_value(
                    mapping, "selection_mode", str, path="training.qualitative_logging"
                ),
            ),
            selected_indices=indices,
        )
        _positive(
            result.every_n_epochs, path="training.qualitative_logging.every_n_epochs"
        )
        _positive(result.num_samples, path="training.qualitative_logging.num_samples")
        if result.selection_mode not in {"random", "fixed_indices"}:
            raise SemanticConfigurationError(
                "training.qualitative_logging.selection_mode must be 'random' or 'fixed_indices'."
            )
        if result.selection_mode == "fixed_indices" and not result.selected_indices:
            raise SemanticConfigurationError(
                "training.qualitative_logging.selected_indices must be non-empty for fixed_indices."
            )
        if result.selection_mode == "random" and result.selected_indices is not None:
            raise SemanticConfigurationError(
                "training.qualitative_logging.selected_indices must be null when "
                "selection_mode is 'random'."
            )
        if result.selected_indices is not None:
            if any(index < 0 for index in result.selected_indices):
                raise SemanticConfigurationError(
                    "training.qualitative_logging.selected_indices must be non-negative."
                )
            if len(set(result.selected_indices)) != len(result.selected_indices):
                raise SemanticConfigurationError(
                    "training.qualitative_logging.selected_indices must be unique."
                )
            if result.num_samples != len(result.selected_indices):
                raise SemanticConfigurationError(
                    "training.qualitative_logging.num_samples must equal the number "
                    "of fixed selected_indices."
                )
        return result


@dataclass(frozen=True, slots=True)
class GANTransitionConfig:
    start_epoch: int


@dataclass(frozen=True, slots=True)
class GANConfig:
    enabled: bool
    target_weight: float
    warmup_epochs: int
    generator_gradient_clip_val: float | None
    discriminator_gradient_clip_val: float | None
    transition: GANTransitionConfig

    @classmethod
    def from_mapping(cls, value: object) -> GANConfig:
        mapping = exact_config_mapping(
            value,
            path="training.gan",
            required_keys=_GAN_KEYS,
        )
        transition = exact_config_mapping(
            require_config_mapping(mapping, "transition", path="training.gan"),
            path="training.gan.transition",
            required_keys=_GAN_TRANSITION_KEYS,
        )
        start_epoch = cast(
            "int",
            require_config_value(
                transition, "start_epoch", int, path="training.gan.transition"
            ),
        )
        generator_clip_raw = require_config_value(
            mapping,
            "generator_gradient_clip_val",
            (float, int, type(None)),
            path="training.gan",
        )
        discriminator_clip_raw = require_config_value(
            mapping,
            "discriminator_gradient_clip_val",
            (float, int, type(None)),
            path="training.gan",
        )
        result = cls(
            enabled=cast(
                "bool",
                require_config_value(mapping, "enabled", bool, path="training.gan"),
            ),
            target_weight=float(
                cast(
                    "float | int",
                    require_config_value(
                        mapping, "target_weight", (float, int), path="training.gan"
                    ),
                )
            ),
            warmup_epochs=cast(
                "int",
                require_config_value(
                    mapping, "warmup_epochs", int, path="training.gan"
                ),
            ),
            generator_gradient_clip_val=(
                None
                if generator_clip_raw is None
                else float(cast("float | int", generator_clip_raw))
            ),
            discriminator_gradient_clip_val=(
                None
                if discriminator_clip_raw is None
                else float(cast("float | int", discriminator_clip_raw))
            ),
            transition=GANTransitionConfig(start_epoch=start_epoch),
        )
        _positive(
            start_epoch, path="training.gan.transition.start_epoch", allow_zero=True
        )
        _positive(
            result.target_weight, path="training.gan.target_weight", allow_zero=True
        )
        _positive(result.warmup_epochs, path="training.gan.warmup_epochs")
        _optional_non_negative(
            result.generator_gradient_clip_val,
            path="training.gan.generator_gradient_clip_val",
        )
        _optional_non_negative(
            result.discriminator_gradient_clip_val,
            path="training.gan.discriminator_gradient_clip_val",
        )
        return result


@dataclass(frozen=True, slots=True)
class BaseTrainingConfig:
    """All task-independent training settings consumed by shared runtime code."""

    trainer: TrainerConfig
    optimizer: OptimizerConfig
    checkpoint: CheckpointConfig
    early_stopping: EarlyStoppingConfig
    lr_monitor: LRMonitorConfig
    qualitative_logging: QualitativeLoggingConfig
    gan: GANConfig
    matmul_precision: str
    allow_tf32: bool

    @classmethod
    def from_validated_task_mapping(cls, value: object) -> BaseTrainingConfig:
        """Parse shared fields from a complete task-owned training mapping.

        The caller is required to exact-validate all task extension fields
        before invoking this method.  Shared nested mappings are projected to
        their public base contracts, then recursively exact-validated here.
        """
        task_mapping = as_config_mapping(value, path="training")
        shared = dict(
            _project_required_mapping(
                task_mapping,
                path="training",
                keys=_BASE_TRAINING_KEYS,
            )
        )
        shared["optimizer"] = _project_required_mapping(
            require_config_mapping(task_mapping, "optimizer", path="training"),
            path="training.optimizer",
            keys=_OPTIMIZER_KEYS,
        )
        gan = require_config_mapping(task_mapping, "gan", path="training")
        shared_gan = dict(
            _project_required_mapping(
                gan,
                path="training.gan",
                keys=_GAN_KEYS,
            )
        )
        shared_gan["transition"] = _project_required_mapping(
            require_config_mapping(gan, "transition", path="training.gan"),
            path="training.gan.transition",
            keys=_GAN_TRANSITION_KEYS,
        )
        shared["gan"] = shared_gan
        return cls.from_mapping(shared)

    @classmethod
    def from_mapping(cls, value: object) -> BaseTrainingConfig:
        mapping = exact_config_mapping(
            value,
            path="training",
            required_keys=_BASE_TRAINING_KEYS,
        )
        trainer = TrainerConfig.from_mapping(
            require_config_mapping(mapping, "trainer", path="training")
        )
        matmul_precision = cast(
            "str",
            require_config_value(mapping, "matmul_precision", str, path="training"),
        )
        if matmul_precision not in _MATMUL_PRECISIONS:
            allowed = ", ".join(sorted(_MATMUL_PRECISIONS))
            raise SemanticConfigurationError(
                f"training.matmul_precision must be one of {allowed}; "
                f"got {matmul_precision!r}."
            )
        result = cls(
            trainer=trainer,
            optimizer=OptimizerConfig.from_mapping(
                mapping, max_epochs=trainer.max_epochs
            ),
            checkpoint=CheckpointConfig.from_mapping(
                require_config_mapping(mapping, "checkpoint", path="training")
            ),
            early_stopping=EarlyStoppingConfig.from_mapping(
                require_config_mapping(mapping, "early_stopping", path="training")
            ),
            lr_monitor=LRMonitorConfig.from_mapping(
                require_config_mapping(mapping, "lr_monitor", path="training")
            ),
            qualitative_logging=QualitativeLoggingConfig.from_mapping(
                require_config_mapping(mapping, "qualitative_logging", path="training")
            ),
            gan=GANConfig.from_mapping(
                require_config_mapping(mapping, "gan", path="training")
            ),
            matmul_precision=matmul_precision,
            allow_tf32=cast(
                "bool",
                require_config_value(mapping, "allow_tf32", bool, path="training"),
            ),
        )
        if result.gan.enabled:
            if result.gan.target_weight <= 0:
                raise SemanticConfigurationError(
                    "training.gan.target_weight must be > 0 when GAN is enabled."
                )
            if result.gan.transition.start_epoch >= result.trainer.max_epochs:
                raise SemanticConfigurationError(
                    "training.gan.transition.start_epoch must be less than "
                    "training.trainer.max_epochs when GAN is enabled."
                )
            if (
                result.gan.transition.start_epoch + result.gan.warmup_epochs
                > result.trainer.max_epochs
            ):
                raise SemanticConfigurationError(
                    "The GAN transition and warmup must finish within "
                    "training.trainer.max_epochs."
                )
            if result.trainer.gradient_clip_val is not None:
                raise SemanticConfigurationError(
                    "training.trainer.gradient_clip_val must be null when GAN is "
                    "enabled; configure the explicit GAN clip values instead."
                )
            if result.early_stopping.enabled:
                raise SemanticConfigurationError(
                    "training.early_stopping.enabled must be false when GAN is enabled."
                )
            optimizer_warmup_epochs = result.optimizer.warmup_epochs
            remaining_epochs = (
                result.trainer.max_epochs - result.gan.transition.start_epoch
            )
            if (
                optimizer_warmup_epochs is not None
                and optimizer_warmup_epochs >= remaining_epochs
                and optimizer_warmup_epochs != 0
            ):
                raise SemanticConfigurationError(
                    "training.warmup_epochs must be 0 or less than the epochs "
                    "remaining after the GAN transition."
                )
        return result


@dataclass(frozen=True, slots=True)
class TrainingRuntimeConfig:
    """Validated shared training boundary, including the common path roots."""

    run: BaseRunConfig
    training: BaseTrainingConfig
    resolver: PathResolver

    @classmethod
    def from_config(
        cls,
        value: object,
        *,
        repository_root: Path,
    ) -> TrainingRuntimeConfig:
        """Validate all shared sections before any training side effect."""
        from src.utils.configuration import RuntimePathRoots

        config = as_config_mapping(value, path="configuration")
        paths = require_config_mapping(config, "paths", path="configuration")
        resolver = PathResolver(
            RuntimePathRoots.from_mapping(paths, repository_root=repository_root)
        )
        return cls(
            run=BaseRunConfig.from_mapping(
                require_config_mapping(config, "run", path="configuration"),
                resolver=resolver,
            ),
            training=BaseTrainingConfig.from_validated_task_mapping(
                require_config_mapping(config, "training", path="configuration")
            ),
            resolver=resolver,
        )


@dataclass(frozen=True, slots=True)
class BaseDataConfig:
    """Shared scene-directory DataLoader contract."""

    scene_dir: Path
    batch_size: int
    num_workers: int
    pin_memory: bool

    @classmethod
    def from_validated_task_mapping(
        cls,
        value: object,
        *,
        resolver: PathResolver,
    ) -> BaseDataConfig:
        """Parse the closed shared projection of an exact task data mapping."""
        return cls.from_mapping(
            _project_required_mapping(value, path="data", keys=_BASE_DATA_KEYS),
            resolver=resolver,
        )

    @classmethod
    def from_mapping(cls, value: object, *, resolver: PathResolver) -> BaseDataConfig:
        mapping = exact_config_mapping(
            value,
            path="data",
            required_keys=_BASE_DATA_KEYS,
        )
        scene_dir = cast(
            "str", require_config_value(mapping, "scene_dir", str, path="data")
        )
        if not scene_dir:
            raise SemanticConfigurationError("data.scene_dir must not be empty.")
        result = cls(
            scene_dir=resolver.resolve(PathRole.DATA, scene_dir),
            batch_size=cast(
                "int", require_config_value(mapping, "batch_size", int, path="data")
            ),
            num_workers=cast(
                "int", require_config_value(mapping, "num_workers", int, path="data")
            ),
            pin_memory=cast(
                "bool", require_config_value(mapping, "pin_memory", bool, path="data")
            ),
        )
        _positive(result.batch_size, path="data.batch_size")
        _positive(result.num_workers, path="data.num_workers", allow_zero=True)
        return result


@dataclass(frozen=True, slots=True)
class ChunkDataConfig:
    """Shared generated-chunk settings, with an artifact-root derived directory."""

    scenes_per_chunk: int
    epochs_per_chunk: int
    prefetch_chunks: int
    chunks_dir: Path
    generation_workers: int
    generator_device: str

    @classmethod
    def from_validated_task_mapping(
        cls,
        value: object,
        *,
        resolver: PathResolver,
    ) -> ChunkDataConfig:
        """Parse the closed chunk projection of an exact task data mapping."""
        task_data = as_config_mapping(value, path="data")
        shared = dict(
            _project_required_mapping(
                task_data,
                path="data",
                keys=_CHUNK_DATA_KEYS,
            )
        )
        shared["chunk"] = _project_required_mapping(
            require_config_mapping(task_data, "chunk", path="data"),
            path="data.chunk",
            keys=_CHUNK_KEYS,
        )
        return cls.from_mapping(shared, resolver=resolver)

    @classmethod
    def from_mapping(cls, value: object, *, resolver: PathResolver) -> ChunkDataConfig:
        data = exact_config_mapping(
            value,
            path="data",
            required_keys=_CHUNK_DATA_KEYS,
        )
        chunk = exact_config_mapping(
            require_config_mapping(data, "chunk", path="data"),
            path="data.chunk",
            required_keys=_CHUNK_KEYS,
        )
        chunks_dir = cast(
            "str", require_config_value(chunk, "chunks_dir", str, path="data.chunk")
        )
        result = cls(
            scenes_per_chunk=cast(
                "int",
                require_config_value(chunk, "scenes_per_chunk", int, path="data.chunk"),
            ),
            epochs_per_chunk=cast(
                "int",
                require_config_value(chunk, "epochs_per_chunk", int, path="data.chunk"),
            ),
            prefetch_chunks=cast(
                "int",
                require_config_value(chunk, "prefetch_chunks", int, path="data.chunk"),
            ),
            chunks_dir=resolver.resolve(PathRole.ARTIFACT, chunks_dir),
            generation_workers=cast(
                "int",
                require_config_value(
                    chunk, "generation_workers", int, path="data.chunk"
                ),
            ),
            generator_device=cast(
                "str", require_config_value(data, "generator_device", str, path="data")
            ),
        )
        for field_name, field_value in (
            ("scenes_per_chunk", result.scenes_per_chunk),
            ("epochs_per_chunk", result.epochs_per_chunk),
            ("generation_workers", result.generation_workers),
        ):
            _positive(field_value, path=f"data.chunk.{field_name}")
        _positive(
            result.prefetch_chunks,
            path="data.chunk.prefetch_chunks",
            allow_zero=True,
        )
        _non_empty(result.generator_device, path="data.generator_device")
        return result


@dataclass(frozen=True, slots=True)
class SceneVisualizationConfig:
    """Shared typed visualization input/output and selection settings."""

    mode: str
    scene_path: Path
    checkpoint: Path | None
    device: str
    animation_view: str
    fps: float | None
    save: Path | None
    camera: int
    cameras: tuple[int, ...] | Literal["all"] | None
    info: bool

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        resolver: PathResolver,
        extension_keys: Set[str] = frozenset(),
    ) -> SceneVisualizationConfig:
        """Validate the complete visualization mapping before any scene I/O.

        Task-specific keys must be named explicitly in ``extension_keys``;
        unlisted old or misspelled keys are rejected before shared fields are
        read.  Task owners remain responsible for validating extension values.
        """
        shared_keys = frozenset(
            {
                "mode",
                "scene_path",
                "checkpoint",
                "device",
                "animation_view",
                "fps",
                "save",
                "camera",
                "cameras",
                "info",
                "style",
                "view_3d",
            }
        )
        mapping = exact_config_mapping(
            value,
            path="visualization",
            required_keys=shared_keys,
            optional_keys=extension_keys,
        )
        mode = cast(
            "str", require_config_value(mapping, "mode", str, path="visualization")
        )
        scene_path = cast(
            "str",
            require_config_value(mapping, "scene_path", str, path="visualization"),
        )
        device = cast(
            "str", require_config_value(mapping, "device", str, path="visualization")
        )
        animation_view = cast(
            "str",
            require_config_value(mapping, "animation_view", str, path="visualization"),
        )
        if any(not value for value in (mode, scene_path, device, animation_view)):
            raise SemanticConfigurationError(
                "visualization mode, scene_path, device, and animation_view must not be empty."
            )
        checkpoint_raw = cast(
            "str | None",
            require_config_value(
                mapping, "checkpoint", (str, type(None)), path="visualization"
            ),
        )
        save_raw = cast(
            "str | None",
            require_config_value(
                mapping, "save", (str, type(None)), path="visualization"
            ),
        )
        if checkpoint_raw == "" or save_raw == "":
            raise SemanticConfigurationError(
                "visualization checkpoint/save must be null or a non-empty path."
            )
        fps_raw = require_config_value(
            mapping, "fps", (float, int, type(None)), path="visualization"
        )
        fps = None if fps_raw is None else float(cast("float | int", fps_raw))
        if fps is not None:
            _positive(fps, path="visualization.fps")

        cameras_raw = require_config_value(
            mapping,
            "cameras",
            (str, list, tuple, type(None)),
            path="visualization",
        )
        cameras: tuple[int, ...] | Literal["all"] | None
        if cameras_raw is None:
            cameras = None
        elif isinstance(cameras_raw, str):
            if cameras_raw == "all":
                cameras = "all"
            elif not cameras_raw or any(
                not part.strip() for part in cameras_raw.split(",")
            ):
                raise SemanticConfigurationError(
                    "visualization.cameras must be 'all', null, or comma-separated integers."
                )
            else:
                try:
                    cameras = tuple(
                        int(part.strip()) for part in cameras_raw.split(",")
                    )
                except ValueError as error:
                    raise SemanticConfigurationError(
                        "visualization.cameras must contain comma-separated integers."
                    ) from error
        else:
            raw_sequence = cast("Sequence[object]", cameras_raw)
            if any(type(item) is not int for item in raw_sequence):
                raise ConfigurationTypeError(
                    "visualization.cameras must contain only exact int values."
                )
            cameras = tuple(cast("int", item) for item in raw_sequence)
        if isinstance(cameras, tuple) and any(camera < 0 for camera in cameras):
            raise SemanticConfigurationError(
                "visualization.cameras indices must be non-negative."
            )
        camera = cast(
            "int", require_config_value(mapping, "camera", int, path="visualization")
        )
        _positive(camera, path="visualization.camera", allow_zero=True)
        return cls(
            mode=mode,
            scene_path=resolver.resolve(PathRole.DATA, scene_path),
            checkpoint=(
                None
                if checkpoint_raw is None
                else resolver.resolve(PathRole.CHECKPOINT, checkpoint_raw)
            ),
            device=device,
            animation_view=animation_view,
            fps=fps,
            save=(
                None
                if save_raw is None
                else resolver.resolve(PathRole.OUTPUT, save_raw)
            ),
            camera=camera,
            cameras=cameras,
            info=cast(
                "bool",
                require_config_value(mapping, "info", bool, path="visualization"),
            ),
        )
