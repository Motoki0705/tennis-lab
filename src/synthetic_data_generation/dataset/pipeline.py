"""Shared immutable contracts for dataset pipeline planning and execution."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Self

from src.utils.io import save_json_atomic

DATASET_PIPELINE_PLAN_SCHEMA = "tennis_synthetic_dataset_pipeline_plan_v1"
PipelineRuntime = Literal["project", "nht"]


def _canonical_fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PipelineCommand:
    """One fixed Python module invocation in a declared runtime."""

    stage: str
    runtime: PipelineRuntime
    module: str
    arguments: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.stage or self.stage.strip() != self.stage:
            raise ValueError("Pipeline stage names must be normalized.")
        if self.runtime not in {"project", "nht"}:
            raise ValueError(f"Unsupported pipeline runtime: {self.runtime!r}.")
        prefix = "src.synthetic_data_generation."
        if not self.module.startswith(prefix):
            raise ValueError(
                f"Pipeline modules must live below {prefix!r}: {self.module!r}."
            )
        arguments = tuple(self.arguments)
        if any(not value or "\x00" in value for value in arguments):
            raise ValueError("Pipeline arguments must be non-empty strings.")
        object.__setattr__(self, "arguments", arguments)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible command record."""
        return {
            "stage": self.stage,
            "runtime": self.runtime,
            "module": self.module,
            "arguments": list(self.arguments),
        }


@dataclass(frozen=True)
class DatasetPipelinePlan:
    """A deterministic, reviewable plan for one dataset pipeline run."""

    dataset: str
    selected_algorithms: Mapping[str, str]
    commands: tuple[PipelineCommand, ...]
    plan_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.dataset or self.dataset.strip() != self.dataset:
            raise ValueError("Dataset names must be non-empty and normalized.")
        algorithms = dict(self.selected_algorithms)
        if not algorithms or any(
            not key or not value for key, value in algorithms.items()
        ):
            raise ValueError("A dataset plan requires named algorithm selections.")
        commands = tuple(self.commands)
        if not commands:
            raise ValueError("A dataset plan requires at least one command.")
        stages = [command.stage for command in commands]
        if len(stages) != len(set(stages)):
            raise ValueError(f"Pipeline stages must be unique: {stages}.")
        object.__setattr__(self, "selected_algorithms", algorithms)
        object.__setattr__(self, "commands", commands)
        computed = _canonical_fingerprint(self._unsigned_dict())
        if self.plan_fingerprint and self.plan_fingerprint != computed:
            raise ValueError(
                "Dataset pipeline plan fingerprint differs from its contents."
            )
        object.__setattr__(self, "plan_fingerprint", computed)

    def _unsigned_dict(self) -> dict[str, object]:
        return {
            "schema": DATASET_PIPELINE_PLAN_SCHEMA,
            "dataset": self.dataset,
            "selected_algorithms": dict(sorted(self.selected_algorithms.items())),
            "commands": [command.to_dict() for command in self.commands],
        }

    def to_dict(self) -> dict[str, object]:
        """Return the complete JSON-compatible pipeline plan."""
        return {
            **self._unsigned_dict(),
            "plan_fingerprint": self.plan_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse and strictly validate one v1 pipeline plan."""
        if not isinstance(value, dict):
            raise TypeError("Dataset pipeline plan must be an object.")
        required = {
            "schema",
            "dataset",
            "selected_algorithms",
            "commands",
            "plan_fingerprint",
        }
        if set(value) != required:
            raise ValueError("Dataset pipeline plan fields differ from v1.")
        if value["schema"] != DATASET_PIPELINE_PLAN_SCHEMA:
            raise ValueError("Unsupported dataset pipeline plan schema.")
        raw_algorithms = value["selected_algorithms"]
        raw_commands = value["commands"]
        if not isinstance(raw_algorithms, dict) or not isinstance(raw_commands, list):
            raise TypeError("Pipeline algorithms/commands have invalid types.")
        algorithms: dict[str, str] = {}
        for key, item in raw_algorithms.items():
            if not isinstance(key, str) or not isinstance(item, str):
                raise TypeError("Pipeline algorithm selections must be strings.")
            algorithms[key] = item
        commands: list[PipelineCommand] = []
        for raw in raw_commands:
            if not isinstance(raw, dict) or set(raw) != {
                "stage",
                "runtime",
                "module",
                "arguments",
            }:
                raise ValueError("Pipeline command fields differ from v1.")
            arguments = raw["arguments"]
            if not isinstance(arguments, list) or not all(
                isinstance(item, str) for item in arguments
            ):
                raise TypeError("Pipeline command arguments must be strings.")
            runtime = raw["runtime"]
            if runtime not in {"project", "nht"}:
                raise ValueError(f"Unsupported pipeline runtime: {runtime!r}.")
            stage = raw["stage"]
            module = raw["module"]
            fingerprint = value["plan_fingerprint"]
            dataset = value["dataset"]
            if (
                not isinstance(stage, str)
                or not isinstance(module, str)
                or not isinstance(fingerprint, str)
                or not isinstance(dataset, str)
            ):
                raise TypeError("Pipeline command or identity fields are invalid.")
            commands.append(
                PipelineCommand(
                    stage=stage,
                    runtime=runtime,
                    module=module,
                    arguments=tuple(arguments),
                )
            )
        return cls(
            dataset=dataset,
            selected_algorithms=algorithms,
            commands=tuple(commands),
            plan_fingerprint=fingerprint,
        )

    def write(self, path: Path) -> None:
        """Atomically publish the pipeline plan."""
        save_json_atomic(self.to_dict(), path)


class DatasetPipeline(Protocol):
    """Interface implemented by every dataset-specific pipeline."""

    @property
    def dataset_name(self) -> str:
        """Return the registry key for the dataset."""

    def build_plan(self, config: Mapping[str, object]) -> DatasetPipelinePlan:
        """Validate configuration and build an immutable execution plan."""


def require_mapping(
    value: object,
    *,
    name: str,
) -> Mapping[str, object]:
    """Return a string-keyed mapping or raise a precise configuration error."""
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping.")
    return value


def require_string(value: object, *, name: str) -> str:
    """Return a non-empty string or raise a precise configuration error."""
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string.")
    return value


def command_arguments(
    stages: Mapping[str, object],
    *,
    stage: str,
) -> tuple[str, ...] | None:
    """Read an explicitly enabled stage's literal argv list."""
    raw = stages.get(stage)
    if raw is None:
        return None
    stage_config = require_mapping(raw, name=f"stages.{stage}")
    enabled = stage_config.get("enabled")
    if not isinstance(enabled, bool):
        raise TypeError(f"stages.{stage}.enabled must be boolean.")
    if not enabled:
        return None
    arguments = stage_config.get("arguments")
    if not isinstance(arguments, Sequence) or isinstance(arguments, (str, bytes)):
        raise TypeError(f"stages.{stage}.arguments must be a sequence.")
    if not all(isinstance(item, str) and item for item in arguments):
        raise TypeError(f"stages.{stage}.arguments must contain non-empty strings.")
    return tuple(arguments)


def configured_command(
    stages: Mapping[str, object],
    *,
    stage: str,
    runtime: PipelineRuntime,
    module: str,
) -> PipelineCommand | None:
    """Build a command when a stage is explicitly enabled."""
    arguments = command_arguments(stages, stage=stage)
    if arguments is None:
        return None
    return PipelineCommand(
        stage=stage,
        runtime=runtime,
        module=module,
        arguments=arguments,
    )
