"""Strict update-count scheduling independent of dataset length and epochs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from src.tasks.base.configuration import as_config_mapping, require_config_mapping
from src.utils.configuration import SemanticConfigurationError


@dataclass(frozen=True, slots=True)
class StepSchedule:
    max_steps: int
    validation_every_n_steps: int
    checkpoint_every_n_steps: int

    @classmethod
    def from_config(cls, value: object) -> StepSchedule:
        config = as_config_mapping(value, path="configuration")
        training = require_config_mapping(config, "training", path="configuration")
        schedule = require_config_mapping(training, "step_schedule", path="training")
        keys = {"max_steps", "validation_every_n_steps", "checkpoint_every_n_steps"}
        if set(schedule) != keys or any(
            type(schedule[key]) is not int or schedule[key] <= 0 for key in keys
        ):
            raise SemanticConfigurationError(
                "step_schedule requires positive integer max_steps, validation_every_n_steps and checkpoint_every_n_steps."
            )
        result = cls(**dict(schedule))
        if (
            result.max_steps % result.validation_every_n_steps
            or result.max_steps % result.checkpoint_every_n_steps
        ):
            raise SemanticConfigurationError(
                "max_steps must be divisible by validation and checkpoint intervals."
            )
        if (
            training["warmup_epochs"] is not None
            or training["steps_per_epoch"] is not None
        ):
            raise SemanticConfigurationError(
                "Step schedule forbids epoch warmup and steps_per_epoch overrides."
            )
        warmup = training["warmup_steps"]
        if type(warmup) is not int or not 0 <= warmup < result.max_steps:
            raise SemanticConfigurationError(
                "warmup_steps must be an integer in [0, max_steps)."
            )
        trainer = require_config_mapping(training, "trainer", path="training")
        if trainer["max_epochs"] != 1 or trainer["check_val_every_n_epoch"] != 1:
            raise SemanticConfigurationError(
                "Step mode requires legacy epoch fields fixed at their inert value 1."
            )
        if trainer["accumulate_grad_batches"] != 1:
            raise SemanticConfigurationError(
                "Step mode currently requires accumulate_grad_batches=1."
            )
        for section in ("checkpoint", "early_stopping", "qualitative_logging", "gan"):
            if require_config_mapping(training, section, path="training")["enabled"]:
                raise SemanticConfigurationError(
                    f"Step mode forbids the epoch-based {section} callback."
                )
        run = require_config_mapping(config, "run", path="configuration")
        if run["test_after_fit"] or run["fast_dev_run"]:
            raise SemanticConfigurationError(
                "Step mode requires explicit final test evaluation and a full update-count run."
            )
        return result


def has_step_schedule(config: object) -> bool:
    mapping = as_config_mapping(config, path="configuration")
    training = mapping.get("training")
    return isinstance(training, Mapping) and "step_schedule" in training
