"""Typed contracts shared by every canonical scene-pipeline stage."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

if TYPE_CHECKING:
    from src.synthetic_data_generation.pipeline.workspace import SceneWorkspace

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


class StageInputKind(StrEnum):
    """Finite authorities a stage may require before mutation or execution."""

    SOURCE_VIDEO = "source_video"
    RESOLVED_CONFIGURATION = "resolved_configuration"
    STAGE_OUTPUT = "stage_output"


@dataclass(frozen=True, slots=True)
class ScenePipelineRequest:
    """One strict request with explicit start, terminal, and dataset stages."""

    scene_id: str
    source_video: Path
    targets: frozenset[DatasetTarget]
    from_stage: StageName
    through_stage: StageName
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
        if not isinstance(self.from_stage, StageName):
            raise TypeError("from_stage must be a StageName value.")
        if not isinstance(self.through_stage, StageName):
            raise TypeError("through_stage must be a StageName value.")
        terminal_target = next(
            (
                target
                for target in DatasetTarget
                if target.stage is self.through_stage
            ),
            None,
        )
        if terminal_target is not None and terminal_target not in self.targets:
            raise ValueError(
                f"through_stage {self.through_stage.value!r} requires the "
                f"{terminal_target.value!r} dataset target."
            )
        if not self.config_schema.strip() or self.config_schema != self.config_schema.strip():
            raise ValueError("config_schema must be a non-empty trimmed identifier.")

    @property
    def active_targets(self) -> frozenset[DatasetTarget]:
        """Return dataset targets inside the terminal stage dependency closure."""
        if self.through_stage is StageName.REPORT:
            return self.targets
        return frozenset(
            target for target in self.targets if target.stage is self.through_stage
        )

    def to_dict(self) -> dict[str, object]:
        """Return stable semantic request fields without an identity digest."""
        return {
            "scene_id": self.scene_id,
            "source_video": str(self.source_video),
            "targets": sorted(target.value for target in self.targets),
            "from_stage": self.from_stage.value,
            "through_stage": self.through_stage.value,
            "config_schema": self.config_schema,
        }


@dataclass(frozen=True, slots=True)
class StageInput:
    """One typed request, configuration, or upstream-artifact requirement."""

    kind: StageInputKind
    producer: StageName | None = None
    relative_path: Path | None = None
    target: DatasetTarget | None = None

    def __post_init__(self) -> None:
        stage_bound = self.kind is StageInputKind.STAGE_OUTPUT
        if stage_bound != (self.producer is not None and self.relative_path is not None):
            raise ValueError(
                "Stage-output inputs require both producer and relative_path; "
                "request/configuration inputs require neither."
            )
        if self.relative_path is not None:
            _validate_relative_path(self.relative_path, label="Stage input")
        if self.target is not None and not stage_bound:
            raise ValueError("Only stage-output inputs may be target-conditional.")

    @classmethod
    def source_video(cls) -> StageInput:
        """Return the typed request source-video authority."""
        return cls(StageInputKind.SOURCE_VIDEO)

    @classmethod
    def resolved_configuration(cls) -> StageInput:
        """Return the typed resolved-configuration authority."""
        return cls(StageInputKind.RESOLVED_CONFIGURATION)

    @classmethod
    def stage_output(
        cls,
        producer: StageName,
        relative_path: str | Path,
        *,
        target: DatasetTarget | None = None,
    ) -> StageInput:
        """Return a typed upstream artifact requirement."""
        return cls(
            StageInputKind.STAGE_OUTPUT,
            producer=producer,
            relative_path=Path(relative_path),
            target=target,
        )

    def applies_to(self, request: ScenePipelineRequest) -> bool:
        """Return whether this input is required for ``request``."""
        return self.target is None or self.target in request.targets


@dataclass(frozen=True, slots=True)
class StageExecutionSummary:
    """JSON-safe semantic completion summary persisted in ``run.json``."""

    values: Mapping[str, object]

    def __post_init__(self) -> None:
        copied = dict(self.values)
        if any(not isinstance(key, str) or not key for key in copied):
            raise TypeError("Stage summary keys must be non-empty strings.")
        _validate_json_value(copied, path="summary")
        object.__setattr__(self, "values", MappingProxyType(copied))


SummaryT = TypeVar("SummaryT", bound=StageExecutionSummary)
SummaryT_co = TypeVar("SummaryT_co", bound=StageExecutionSummary, covariant=True)


class StageExecutionContext(Protocol):
    """Minimal context visible to handlers without runner internals."""

    @property
    def request(self) -> ScenePipelineRequest: ...

    @property
    def stage(self) -> StageDefinition[StageExecutionSummary]: ...

    @property
    def owner_path(self) -> Path: ...

    @property
    def staging_path(self) -> Path: ...


class StageHandler(Protocol[SummaryT_co]):
    """One modular stage owner with explicit lifecycle operations."""

    def preflight(self, context: StageExecutionContext) -> None: ...

    def execute(self, context: StageExecutionContext) -> SummaryT_co: ...

    def validate(self, context: StageExecutionContext) -> None: ...


@dataclass(frozen=True, slots=True)
class StagePublicationResult:
    """Typed evidence that publication resolved to the one fixed owner path."""

    owner_path: Path
    replaced_existing: bool


class StagePublicationStrategy(Protocol):
    """A complete publication lifecycle bound into one stage definition."""

    def preflight(
        self,
        workspace: SceneWorkspace,
        definition: StageDefinition[StageExecutionSummary],
    ) -> None: ...

    def prepare(
        self,
        workspace: SceneWorkspace,
        definition: StageDefinition[StageExecutionSummary],
    ) -> Path: ...

    def publish(
        self,
        workspace: SceneWorkspace,
        definition: StageDefinition[StageExecutionSummary],
    ) -> StagePublicationResult: ...

    def recover(
        self,
        workspace: SceneWorkspace,
        definition: StageDefinition[StageExecutionSummary],
    ) -> None: ...

    def abandon(
        self,
        workspace: SceneWorkspace,
        definition: StageDefinition[StageExecutionSummary],
    ) -> None: ...

    def invalidate(
        self,
        workspace: SceneWorkspace,
        definition: StageDefinition[StageExecutionSummary],
    ) -> None: ...


class ReusablePublicationValidator(Protocol):
    """One stage's typed semantic gate for an already-completed owner."""

    def validate(self, owner_path: Path) -> None:
        """Raise when the fixed owner is not safe to reuse."""
        ...


@dataclass(frozen=True, slots=True)
class StageDefinition(Generic[SummaryT]):
    """The sole typed authority for one stage's graph and full lifecycle."""

    name: StageName
    dependencies: tuple[StageName, ...]
    owner_relative_path: Path
    required_inputs: tuple[StageInput, ...]
    required_outputs: tuple[Path, ...]
    handler: StageHandler[SummaryT]
    publication: StagePublicationStrategy
    reusable_publication_validator: ReusablePublicationValidator
    summary_type: type[SummaryT]
    _descendants: tuple[StageName, ...] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.name in self.dependencies:
            raise ValueError(f"Stage {self.name.value} cannot depend on itself.")
        if len(self.dependencies) != len(set(self.dependencies)):
            raise ValueError(f"Stage {self.name.value} has duplicate dependencies.")
        _validate_relative_path(self.owner_relative_path, label="Stage owner")
        if not self.required_inputs:
            raise ValueError(f"Stage {self.name.value} must declare required inputs.")
        if len(self.required_inputs) != len(set(self.required_inputs)):
            raise ValueError(f"Stage {self.name.value} has duplicate required inputs.")
        if not self.required_outputs:
            raise ValueError(f"Stage {self.name.value} must declare required outputs.")
        if len(self.required_outputs) != len(set(self.required_outputs)):
            raise ValueError(f"Stage {self.name.value} has duplicate required outputs.")
        for output in self.required_outputs:
            _validate_relative_path(output, label="Stage output")
        for method in ("preflight", "execute", "validate"):
            if not callable(getattr(self.handler, method, None)):
                raise TypeError(f"Stage {self.name.value} has an unbound handler lifecycle.")
        for method in (
            "preflight",
            "prepare",
            "publish",
            "recover",
            "abandon",
            "invalidate",
        ):
            if not callable(getattr(self.publication, method, None)):
                raise TypeError(
                    f"Stage {self.name.value} has an incomplete publication strategy."
                )
        if not callable(
            getattr(self.reusable_publication_validator, "validate", None)
        ):
            raise TypeError(
                f"Stage {self.name.value} has no reusable-publication validator."
            )
        if not issubclass(self.summary_type, StageExecutionSummary):
            raise TypeError("Stage summary_type must derive from StageExecutionSummary.")

    @property
    def descendants(self) -> tuple[StageName, ...]:
        """Return graph-derived descendants after registry validation."""
        if self._descendants is None:
            raise RuntimeError("Stage definition is not bound to a validated registry.")
        return self._descendants

    def _bind_descendants(self, descendants: tuple[StageName, ...]) -> None:
        if self._descendants is not None:
            raise ValueError(f"Stage {self.name.value} is already bound to a registry.")
        object.__setattr__(self, "_descendants", descendants)

    def preflight(self, context: StageExecutionContext) -> None:
        """Run the bound handler preflight."""
        self.handler.preflight(context)

    def execute(self, context: StageExecutionContext) -> SummaryT:
        """Run the bound handler and enforce this stage's summary type."""
        summary = self.handler.execute(context)
        if not isinstance(summary, self.summary_type):
            raise TypeError(
                f"Stage {self.name.value} returned {type(summary).__name__}; "
                f"expected {self.summary_type.__name__}."
            )
        return summary

    def validate(self, context: StageExecutionContext) -> None:
        """Run the bound semantic validator."""
        self.handler.validate(context)

    def preflight_publication(self, workspace: SceneWorkspace) -> None:
        """Verify publication authority before destructive invalidation."""
        self.publication.preflight(workspace, _as_execution_definition(self))

    def prepare_publication(self, workspace: SceneWorkspace) -> Path:
        """Prepare this definition's fixed transaction location."""
        return self.publication.prepare(workspace, _as_execution_definition(self))

    def publish(self, workspace: SceneWorkspace) -> StagePublicationResult:
        """Publish the validated owner snapshot through the bound strategy."""
        return self.publication.publish(workspace, _as_execution_definition(self))

    def recover_publication(self, workspace: SceneWorkspace) -> None:
        """Recover one interrupted publication without selecting a fallback."""
        self.publication.recover(workspace, _as_execution_definition(self))

    def abandon_publication(self, workspace: SceneWorkspace) -> None:
        """Clear partial transaction state after a failed attempt."""
        self.publication.abandon(workspace, _as_execution_definition(self))

    def invalidate_publication(self, workspace: SceneWorkspace) -> None:
        """Remove this stage's canonical owner and transaction residue."""
        self.publication.invalidate(workspace, _as_execution_definition(self))

    def validate_reusable_publication(self, owner_path: Path) -> None:
        """Apply the bound semantic reuse gate to the fixed owner."""
        self.reusable_publication_validator.validate(owner_path)


@dataclass(frozen=True, slots=True)
class StageExecutionPlan:
    """One graph-derived, publication-aware plan for every runner lifecycle phase."""

    selected: tuple[StageDefinition[StageExecutionSummary], ...]
    cursor: StageDefinition[StageExecutionSummary]
    retained_ancestors: tuple[StageDefinition[StageExecutionSummary], ...]
    invalidated: tuple[StageDefinition[StageExecutionSummary], ...]
    execution: tuple[StageDefinition[StageExecutionSummary], ...]

    def __post_init__(self) -> None:
        inventories = {
            "selected": self.selected,
            "retained_ancestors": self.retained_ancestors,
            "invalidated": self.invalidated,
            "execution": self.execution,
        }
        names: dict[str, set[StageName]] = {}
        for label, definitions in inventories.items():
            inventory_names = tuple(definition.name for definition in definitions)
            if len(inventory_names) != len(set(inventory_names)):
                raise ValueError(f"Execution plan {label} contains duplicate stages.")
            names[label] = set(inventory_names)
        if self.cursor.name not in names["selected"]:
            raise ValueError("Execution-plan cursor must belong to the selected request stages.")
        if self.cursor.name not in names["invalidated"]:
            raise ValueError("Execution-plan cursor must be invalidated before execution.")
        if self.cursor.name not in names["execution"]:
            raise ValueError("Execution-plan cursor must belong to the execution stages.")
        if not names["retained_ancestors"] <= names["selected"]:
            raise ValueError(
                "Retained prerequisites must belong to the selected request stages."
            )
        if names["retained_ancestors"] & names["invalidated"]:
            raise ValueError("Retained prerequisites cannot also be invalidated.")
        if not names["execution"] <= names["selected"]:
            raise ValueError("Execution stages must belong to the selected request stages.")
        if not names["execution"] <= names["invalidated"]:
            raise ValueError("Execution stages must be invalidated before they begin.")


def _as_execution_definition(
    definition: StageDefinition[SummaryT],
) -> StageDefinition[StageExecutionSummary]:
    # StageExecutionSummary is covariant at the handler boundary and publication never
    # consumes a concrete summary. Keeping this conversion private avoids Any in the API.
    return definition  # type: ignore[return-value]


def _validate_relative_path(path: Path, *, label: str) -> None:
    if path.is_absolute() or ".." in path.parts or path == Path(".") or not path.parts:
        raise ValueError(f"{label} paths must be non-empty workspace-relative paths.")


def _validate_json_value(value: object, *, path: str) -> None:
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError(f"{path} contains a non-finite float.")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key.")
            _validate_json_value(nested, path=f"{path}.{key}")
        return
    if isinstance(value, list | tuple):
        for index, nested in enumerate(value):
            _validate_json_value(nested, path=f"{path}[{index}]")
        return
    raise TypeError(f"{path} contains unsupported value type {type(value).__name__}.")


__all__ = [
    "DatasetTarget",
    "ReusablePublicationValidator",
    "ScenePipelineRequest",
    "StageDefinition",
    "StageExecutionContext",
    "StageExecutionPlan",
    "StageExecutionSummary",
    "StageHandler",
    "StageInput",
    "StageInputKind",
    "StageName",
    "StagePublicationResult",
    "StagePublicationStrategy",
    "StageStatus",
]
