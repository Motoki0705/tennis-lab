"""Typed contracts shared by every canonical scene-pipeline stage."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

_SCENE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class StageName(StrEnum):
    """The complete canonical scene stage vocabulary."""

    INGEST = "ingest"
    RECONSTRUCTION = "reconstruction"
    ALIGNMENT = "alignment"
    COURT_DATASET = "court_dataset"
    BLCS_DATASET = "blcs_dataset"
    PLCS_DATASET = "plcs_dataset"
    REPORT = "report"


class DatasetTarget(StrEnum):
    """Dataset domains that may be explicitly requested."""

    COURT = "court"
    BLCS = "blcs"
    PLCS = "plcs"

    @property
    def stage(self) -> StageName:
        """Return the one stage owned by this target."""
        return {
            DatasetTarget.COURT: StageName.COURT_DATASET,
            DatasetTarget.BLCS: StageName.BLCS_DATASET,
            DatasetTarget.PLCS: StageName.PLCS_DATASET,
        }[self]


class StageStatus(StrEnum):
    """Observable lifecycle states stored in the single mutable run manifest."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INVALIDATED = "invalidated"
    SKIPPED = "skipped"


class PublicationMode(StrEnum):
    """How one handler exposes validated output at its fixed owner path."""

    ATOMIC_OUTPUTS = "atomic_outputs"
    EXTERNAL_ATOMIC = "external_atomic"


@dataclass(frozen=True, slots=True)
class ScenePipelineRequest:
    """One strict request for a scene and its explicit dataset targets."""

    scene_id: str
    source_video: Path
    targets: frozenset[DatasetTarget]
    from_stage: StageName
    config_schema: str

    def __post_init__(self) -> None:
        if _SCENE_ID.fullmatch(self.scene_id) is None:
            raise ValueError(f"scene_id is not a portable fixed-path identifier: {self.scene_id!r}.")
        if not self.source_video.is_absolute():
            raise ValueError("source_video must be an absolute path resolved at the boundary.")
        if not self.source_video.is_file():
            raise FileNotFoundError(f"source_video does not exist: {self.source_video}")
        if not self.targets:
            raise ValueError("At least one explicit dataset target is required.")
        if any(not isinstance(target, DatasetTarget) for target in self.targets):
            raise TypeError("targets must contain only DatasetTarget values.")
        if not self.config_schema.strip() or self.config_schema != self.config_schema.strip():
            raise ValueError("config_schema must be a non-empty trimmed identifier.")

    def to_dict(self) -> dict[str, object]:
        """Return stable semantic request fields without an identity digest."""
        return {
            "scene_id": self.scene_id,
            "source_video": str(self.source_video),
            "targets": sorted(target.value for target in self.targets),
            "from_stage": self.from_stage.value,
            "config_schema": self.config_schema,
        }


@dataclass(frozen=True, slots=True)
class StageSpec:
    """One stage's graph, owner, validation, and publication definition."""

    name: StageName
    dependencies: tuple[StageName, ...]
    owner_relative_path: Path
    required_outputs: tuple[Path, ...]
    publication_mode: PublicationMode
    handler_key: str

    def __post_init__(self) -> None:
        if self.name in self.dependencies:
            raise ValueError(f"Stage {self.name.value} cannot depend on itself.")
        if len(self.dependencies) != len(set(self.dependencies)):
            raise ValueError(f"Stage {self.name.value} has duplicate dependencies.")
        if self.owner_relative_path.is_absolute() or ".." in self.owner_relative_path.parts:
            raise ValueError("Stage owner paths must stay relative to the scene workspace.")
        if not self.owner_relative_path.parts:
            raise ValueError("Stage owner paths must not be empty.")
        if not self.required_outputs:
            raise ValueError(f"Stage {self.name.value} must declare required outputs.")
        for output in self.required_outputs:
            if output.is_absolute() or ".." in output.parts or output == Path("."):
                raise ValueError("Stage output paths must be non-empty owner-relative paths.")
        if not self.handler_key.strip() or self.handler_key != self.handler_key.strip():
            raise ValueError("handler_key must be a non-empty trimmed string.")


@dataclass(frozen=True, slots=True)
class StageExecutionSummary:
    """Semantic completion summary persisted in run.json."""

    values: Mapping[str, object]


class StageExecutionContext(Protocol):
    """Minimal context visible to handlers without runner internals."""

    @property
    def request(self) -> ScenePipelineRequest: ...

    @property
    def stage(self) -> StageSpec: ...

    @property
    def owner_path(self) -> Path: ...

    @property
    def staging_path(self) -> Path: ...


class StageHandler(Protocol):
    """One stage owner with explicit preflight, execution, and validation."""

    def preflight(self, context: StageExecutionContext) -> None: ...

    def execute(self, context: StageExecutionContext) -> StageExecutionSummary: ...

    def validate(self, context: StageExecutionContext) -> None: ...


__all__ = [
    "DatasetTarget",
    "PublicationMode",
    "ScenePipelineRequest",
    "StageExecutionContext",
    "StageExecutionSummary",
    "StageHandler",
    "StageName",
    "StageSpec",
    "StageStatus",
]
