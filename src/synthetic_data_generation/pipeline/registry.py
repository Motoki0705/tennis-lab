"""The sole typed stage graph and handler registry definition."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from src.synthetic_data_generation.pipeline.contracts import (
    PublicationMode,
    ScenePipelineRequest,
    StageName,
    StageSpec,
)


@dataclass(frozen=True, slots=True)
class StageRegistry:
    """Validated stage definitions with graph-derived ordering and descendants."""

    specs: Mapping[StageName, StageSpec]

    def __post_init__(self) -> None:
        if set(self.specs) != set(StageName):
            missing = set(StageName) - set(self.specs)
            unknown = set(self.specs) - set(StageName)
            raise ValueError(f"Stage registry mismatch; missing={missing}, unknown={unknown}.")
        for name, spec in self.specs.items():
            if spec.name is not name:
                raise ValueError(f"Stage registry key disagrees with spec: {name.value}.")
            unknown_dependencies = set(spec.dependencies) - set(self.specs)
            if unknown_dependencies:
                raise ValueError(f"Unknown dependencies for {name.value}: {unknown_dependencies}.")
        self.ordered(set(StageName))

    def spec(self, stage: StageName) -> StageSpec:
        """Return the one definition for ``stage``."""
        return self.specs[stage]

    def ordered(self, selected: Iterable[StageName]) -> tuple[StageName, ...]:
        """Topologically order a selected stage subgraph and reject cycles."""
        selected_set = set(selected)
        result: list[StageName] = []
        visiting: set[StageName] = set()
        visited: set[StageName] = set()

        def visit(stage: StageName) -> None:
            if stage in visited or stage not in selected_set:
                return
            if stage in visiting:
                raise ValueError(f"Cycle in canonical stage graph at {stage.value}.")
            visiting.add(stage)
            for dependency in self.spec(stage).dependencies:
                visit(dependency)
            visiting.remove(stage)
            visited.add(stage)
            result.append(stage)

        for stage in StageName:
            visit(stage)
        if set(result) != selected_set:
            raise ValueError("Selected stage set contains an unknown stage.")
        return tuple(result)

    def selected_for_request(self, request: ScenePipelineRequest) -> tuple[StageName, ...]:
        """Return infrastructure, explicit targets, and report only."""
        selected = {
            StageName.INGEST,
            StageName.RECONSTRUCTION,
            StageName.ALIGNMENT,
            StageName.REPORT,
            *(target.stage for target in request.targets),
        }
        return self.ordered(selected)

    def descendants(self, stage: StageName, *, include_self: bool = False) -> tuple[StageName, ...]:
        """Derive all transitive descendants from direct dependencies."""
        found: set[StageName] = {stage} if include_self else set()
        frontier = [stage]
        while frontier:
            current = frontier.pop()
            for candidate, spec in self.specs.items():
                if current in spec.dependencies and candidate not in found:
                    found.add(candidate)
                    frontier.append(candidate)
        return self.ordered(found)


def canonical_registry() -> StageRegistry:
    """Build the canonical fixed stage graph in one inspectable location."""
    specs = {
        StageName.INGEST: StageSpec(
            name=StageName.INGEST,
            dependencies=(),
            owner_relative_path=Path("source"),
            required_outputs=(Path("video.mp4"), Path("metadata.json")),
            publication_mode=PublicationMode.ATOMIC_OUTPUTS,
            handler_key="ingest",
        ),
        StageName.RECONSTRUCTION: StageSpec(
            name=StageName.RECONSTRUCTION,
            dependencies=(StageName.INGEST,),
            owner_relative_path=Path("reconstruction"),
            required_outputs=(
                Path("run.json"),
                Path("export/scene.json"),
            ),
            publication_mode=PublicationMode.EXTERNAL_ATOMIC,
            handler_key="nht_reconstruction",
        ),
        StageName.ALIGNMENT: StageSpec(
            name=StageName.ALIGNMENT,
            dependencies=(StageName.RECONSTRUCTION,),
            owner_relative_path=Path("alignment"),
            required_outputs=(
                Path("ground-line-map.npz"),
                Path("court-geometry.json"),
                Path("alignment.json"),
                Path("diagnostics"),
            ),
            publication_mode=PublicationMode.ATOMIC_OUTPUTS,
            handler_key="alignment",
        ),
        StageName.COURT_DATASET: StageSpec(
            name=StageName.COURT_DATASET,
            dependencies=(StageName.ALIGNMENT,),
            owner_relative_path=Path("datasets/court"),
            required_outputs=(Path("dataset.json"), Path("samples"), Path("diagnostics")),
            publication_mode=PublicationMode.ATOMIC_OUTPUTS,
            handler_key="court_dataset",
        ),
        StageName.BLCS_DATASET: StageSpec(
            name=StageName.BLCS_DATASET,
            dependencies=(StageName.ALIGNMENT,),
            owner_relative_path=Path("datasets/blcs"),
            required_outputs=(Path("dataset.json"), Path("samples"), Path("diagnostics")),
            publication_mode=PublicationMode.ATOMIC_OUTPUTS,
            handler_key="blcs_dataset",
        ),
        StageName.PLCS_DATASET: StageSpec(
            name=StageName.PLCS_DATASET,
            dependencies=(StageName.ALIGNMENT,),
            owner_relative_path=Path("datasets/plcs"),
            required_outputs=(
                Path("dataset.json"),
                Path("backgrounds"),
                Path("scenes"),
                Path("diagnostics"),
            ),
            publication_mode=PublicationMode.ATOMIC_OUTPUTS,
            handler_key="plcs_dataset",
        ),
        StageName.REPORT: StageSpec(
            name=StageName.REPORT,
            dependencies=(
                StageName.COURT_DATASET,
                StageName.BLCS_DATASET,
                StageName.PLCS_DATASET,
            ),
            owner_relative_path=Path("report"),
            required_outputs=(Path("index.html"), Path("report.json")),
            publication_mode=PublicationMode.ATOMIC_OUTPUTS,
            handler_key="report",
        ),
    }
    return StageRegistry(specs)


__all__ = ["StageRegistry", "canonical_registry"]
