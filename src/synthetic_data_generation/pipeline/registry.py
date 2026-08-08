"""The sole typed stage-definition inventory and validated graph registry."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from src.synthetic_data_generation.pipeline.contracts import (
    DatasetTarget,
    ScenePipelineRequest,
    StageDefinition,
    StageExecutionPlan,
    StageExecutionSummary,
    StageHandler,
    StageInput,
    StageInputKind,
    StageName,
)
from src.synthetic_data_generation.pipeline.publication import (
    AtomicDirectoryPublication,
    ExternalAtomicPublication,
)


@dataclass(frozen=True, slots=True)
class CanonicalStageHandlers:
    """Typed, exhaustive lifecycle bindings for the seven canonical stages."""

    ingest: StageHandler[StageExecutionSummary]
    reconstruction: StageHandler[StageExecutionSummary]
    alignment: StageHandler[StageExecutionSummary]
    court_dataset: StageHandler[StageExecutionSummary]
    blcs_dataset: StageHandler[StageExecutionSummary]
    plcs_dataset: StageHandler[StageExecutionSummary]
    report: StageHandler[StageExecutionSummary]


@dataclass(frozen=True, slots=True)
class StageRegistry:
    """Validated definitions whose graph relationships are mechanically derived."""

    definitions: Mapping[StageName, StageDefinition[StageExecutionSummary]]

    def __post_init__(self) -> None:
        definitions = dict(self.definitions)
        object.__setattr__(self, "definitions", MappingProxyType(definitions))
        if set(definitions) != set(StageName):
            missing = set(StageName) - set(definitions)
            unknown = set(definitions) - set(StageName)
            raise ValueError(
                f"Stage registry mismatch; missing={missing}, unknown={unknown}."
            )
        handler_ids: set[int] = set()
        for name, definition in definitions.items():
            if definition.name is not name:
                raise ValueError(
                    f"Stage registry key disagrees with definition: {name.value}."
                )
            unknown_dependencies = set(definition.dependencies) - set(definitions)
            if unknown_dependencies:
                raise ValueError(
                    f"Unknown dependencies for {name.value}: {unknown_dependencies}."
                )
            handler_id = id(definition.handler)
            if handler_id in handler_ids:
                raise ValueError("One handler instance cannot own multiple stage definitions.")
            handler_ids.add(handler_id)
        self._validate_owner_uniqueness()
        self.ordered_names(set(StageName))
        self._validate_inputs()
        for definition in definitions.values():
            definition._bind_descendants(
                tuple(
                    descendant.name
                    for descendant in self.descendants(
                        definition.name,
                        include_self=False,
                        bind=False,
                    )
                )
            )

    def definition(
        self,
        stage: StageName,
    ) -> StageDefinition[StageExecutionSummary]:
        """Return the one complete definition for ``stage``."""
        return self.definitions[stage]

    def ordered_names(self, selected: Iterable[StageName]) -> tuple[StageName, ...]:
        """Topologically order a selected stage subgraph and reject cycles."""
        selected_set = set(selected)
        if not selected_set <= set(self.definitions):
            raise ValueError("Selected stage set contains an unknown stage.")
        result: list[StageName] = []
        visiting: set[StageName] = set()
        visited: set[StageName] = set()

        def visit(stage: StageName) -> None:
            if stage in visited or stage not in selected_set:
                return
            if stage in visiting:
                raise ValueError(f"Cycle in canonical stage graph at {stage.value}.")
            visiting.add(stage)
            for dependency in self.definition(stage).dependencies:
                visit(dependency)
            visiting.remove(stage)
            visited.add(stage)
            result.append(stage)

        for stage in StageName:
            visit(stage)
        return tuple(result)

    def ordered(
        self,
        selected: Iterable[StageName],
    ) -> tuple[StageDefinition[StageExecutionSummary], ...]:
        """Return selected definitions in canonical topological order."""
        return tuple(self.definition(stage) for stage in self.ordered_names(selected))

    def selected_for_request(
        self,
        request: ScenePipelineRequest,
    ) -> tuple[StageDefinition[StageExecutionSummary], ...]:
        """Return infrastructure, explicit targets, and report definitions only."""
        selected = {
            StageName.INGEST,
            StageName.RECONSTRUCTION,
            StageName.ALIGNMENT,
            StageName.REPORT,
            *(target.stage for target in request.targets),
        }
        return self.ordered(selected)

    def execution_for_request(self, request: ScenePipelineRequest) -> StageExecutionPlan:
        """Build the sole cursor-aware plan for all runner lifecycle phases."""
        selected = self.selected_for_request(request)
        selected_names = {definition.name for definition in selected}
        if request.from_stage not in selected_names:
            requested = ", ".join(sorted(target.value for target in request.targets))
            raise ValueError(
                f"from_stage {request.from_stage.value!r} is not selected by "
                f"request targets {{{requested}}}."
            )
        cursor = self.definition(request.from_stage)
        retained_ancestors = self.ordered(
            selected_names & self._ancestors(request.from_stage)
        )
        invalidated = self.descendants(request.from_stage, include_self=True)
        invalidated_names = {definition.name for definition in invalidated}
        execution = self.ordered(selected_names & invalidated_names)
        return StageExecutionPlan(
            selected=selected,
            cursor=cursor,
            retained_ancestors=retained_ancestors,
            invalidated=invalidated,
            execution=execution,
        )

    def descendants(
        self,
        stage: StageName,
        *,
        include_self: bool = False,
        bind: bool = True,
    ) -> tuple[StageDefinition[StageExecutionSummary], ...]:
        """Derive all transitive descendants from direct dependencies."""
        found: set[StageName] = {stage} if include_self else set()
        frontier = [stage]
        while frontier:
            current = frontier.pop()
            for candidate, definition in self.definitions.items():
                if current in definition.dependencies and candidate not in found:
                    found.add(candidate)
                    frontier.append(candidate)
        definitions = self.ordered(found)
        if bind:
            expected = tuple(definition.name for definition in definitions)
            actual = (
                (stage, *self.definition(stage).descendants)
                if include_self
                else self.definition(stage).descendants
            )
            if tuple(actual) != expected:
                raise RuntimeError("Bound descendant inventory disagrees with the stage graph.")
        return definitions

    def _validate_owner_uniqueness(self) -> None:
        definitions = tuple(self.definitions.values())
        for index, left in enumerate(definitions):
            for right in definitions[index + 1 :]:
                if _paths_overlap(
                    left.owner_relative_path,
                    right.owner_relative_path,
                ):
                    raise ValueError(
                        "Stage owner collision: "
                        f"{left.name.value}={left.owner_relative_path}, "
                        f"{right.name.value}={right.owner_relative_path}."
                    )

    def _validate_inputs(self) -> None:
        for definition in self.definitions.values():
            ancestors = self._ancestors(definition.name)
            for stage_input in definition.required_inputs:
                if stage_input.kind is not StageInputKind.STAGE_OUTPUT:
                    continue
                producer_name = stage_input.producer
                relative_path = stage_input.relative_path
                if producer_name is None or relative_path is None:
                    raise RuntimeError("Invalid StageInput escaped construction validation.")
                if producer_name not in self.definitions:
                    raise ValueError(
                        f"Stage {definition.name.value} has unknown input producer "
                        f"{producer_name.value}."
                    )
                if producer_name not in ancestors:
                    raise ValueError(
                        f"Stage {definition.name.value} input {producer_name.value}/"
                        f"{relative_path} is not produced by an ancestor."
                    )
                outputs = self.definition(producer_name).required_outputs
                if not any(
                    relative_path == output or relative_path.is_relative_to(output)
                    for output in outputs
                ):
                    raise ValueError(
                        f"Stage {definition.name.value} input {producer_name.value}/"
                        f"{relative_path} is not a declared producer output."
                    )
                if (
                    stage_input.target is not None
                    and stage_input.target.stage is not producer_name
                ):
                    raise ValueError(
                        "A target-conditional input must reference that target's stage."
                    )

    def _ancestors(self, stage: StageName) -> set[StageName]:
        found: set[StageName] = set()
        frontier = list(self.definition(stage).dependencies)
        while frontier:
            current = frontier.pop()
            if current in found:
                continue
            found.add(current)
            frontier.extend(self.definition(current).dependencies)
        return found


def canonical_registry(handlers: CanonicalStageHandlers) -> StageRegistry:
    """Bind all graph, path, lifecycle, publication, and summary authority once."""
    config = StageInput.resolved_configuration()
    atomic = AtomicDirectoryPublication()
    external = ExternalAtomicPublication()
    definitions = {
        StageName.INGEST: StageDefinition(
            name=StageName.INGEST,
            dependencies=(),
            owner_relative_path=Path("source"),
            required_inputs=(config, StageInput.source_video()),
            required_outputs=(Path("video.mp4"), Path("metadata.json")),
            handler=handlers.ingest,
            publication=atomic,
            summary_type=StageExecutionSummary,
        ),
        StageName.RECONSTRUCTION: StageDefinition(
            name=StageName.RECONSTRUCTION,
            dependencies=(StageName.INGEST,),
            owner_relative_path=Path("reconstruction"),
            required_inputs=(
                config,
                StageInput.stage_output(StageName.INGEST, "video.mp4"),
            ),
            required_outputs=(Path("run.json"), Path("export")),
            handler=handlers.reconstruction,
            publication=external,
            summary_type=StageExecutionSummary,
        ),
        StageName.ALIGNMENT: StageDefinition(
            name=StageName.ALIGNMENT,
            dependencies=(StageName.RECONSTRUCTION,),
            owner_relative_path=Path("alignment"),
            required_inputs=(
                config,
                StageInput.stage_output(StageName.RECONSTRUCTION, "export/scene.json"),
                StageInput.stage_output(StageName.RECONSTRUCTION, "export/cameras.json"),
                StageInput.stage_output(StageName.RECONSTRUCTION, "export/points_scene.npy"),
                StageInput.stage_output(StageName.RECONSTRUCTION, "export/images"),
                StageInput.stage_output(StageName.RECONSTRUCTION, "export/model"),
            ),
            required_outputs=(
                Path("ground-line-map.npz"),
                Path("court-geometry.json"),
                Path("alignment.json"),
                Path("diagnostics"),
            ),
            handler=handlers.alignment,
            publication=atomic,
            summary_type=StageExecutionSummary,
        ),
        StageName.COURT_DATASET: StageDefinition(
            name=StageName.COURT_DATASET,
            dependencies=(StageName.ALIGNMENT,),
            owner_relative_path=Path("datasets/court"),
            required_inputs=_dataset_inputs(config),
            required_outputs=(Path("dataset.json"), Path("samples"), Path("diagnostics")),
            handler=handlers.court_dataset,
            publication=atomic,
            summary_type=StageExecutionSummary,
        ),
        StageName.BLCS_DATASET: StageDefinition(
            name=StageName.BLCS_DATASET,
            dependencies=(StageName.ALIGNMENT,),
            owner_relative_path=Path("datasets/blcs"),
            required_inputs=_dataset_inputs(config),
            required_outputs=(Path("dataset.json"), Path("samples"), Path("diagnostics")),
            handler=handlers.blcs_dataset,
            publication=atomic,
            summary_type=StageExecutionSummary,
        ),
        StageName.PLCS_DATASET: StageDefinition(
            name=StageName.PLCS_DATASET,
            dependencies=(StageName.ALIGNMENT,),
            owner_relative_path=Path("datasets/plcs"),
            required_inputs=_dataset_inputs(config),
            required_outputs=(
                Path("dataset.json"),
                Path("backgrounds"),
                Path("scenes"),
                Path("diagnostics"),
            ),
            handler=handlers.plcs_dataset,
            publication=atomic,
            summary_type=StageExecutionSummary,
        ),
        StageName.REPORT: StageDefinition(
            name=StageName.REPORT,
            dependencies=(
                StageName.COURT_DATASET,
                StageName.BLCS_DATASET,
                StageName.PLCS_DATASET,
            ),
            owner_relative_path=Path("report"),
            required_inputs=(
                config,
                StageInput.stage_output(StageName.ALIGNMENT, "alignment.json"),
                StageInput.stage_output(
                    StageName.COURT_DATASET,
                    "dataset.json",
                    target=DatasetTarget.COURT,
                ),
                StageInput.stage_output(
                    StageName.BLCS_DATASET,
                    "dataset.json",
                    target=DatasetTarget.BLCS,
                ),
                StageInput.stage_output(
                    StageName.PLCS_DATASET,
                    "dataset.json",
                    target=DatasetTarget.PLCS,
                ),
            ),
            required_outputs=(Path("index.html"), Path("report.json")),
            handler=handlers.report,
            publication=atomic,
            summary_type=StageExecutionSummary,
        ),
    }
    return StageRegistry(definitions)


def _dataset_inputs(config: StageInput) -> tuple[StageInput, ...]:
    return (
        config,
        StageInput.stage_output(StageName.RECONSTRUCTION, "export/scene.json"),
        StageInput.stage_output(StageName.ALIGNMENT, "alignment.json"),
        StageInput.stage_output(StageName.ALIGNMENT, "court-geometry.json"),
    )


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


__all__ = ["CanonicalStageHandlers", "StageRegistry", "canonical_registry"]
