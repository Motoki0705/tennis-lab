"""Typed stage DAG and canonical directory ownership for one scene."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class Stage(StrEnum):
    """Canonical scene stages in topological order."""

    INGEST = "ingest"
    RECONSTRUCTION = "reconstruction"
    ALIGNMENT = "alignment"
    COURT_DATASET = "court_dataset"
    BLCS_DATASET = "blcs_dataset"
    PLCS_DATASET = "plcs_dataset"
    REPORT = "report"


class Target(StrEnum):
    """Requested domain dataset targets."""

    COURT = "court"
    BLCS = "blcs"
    PLCS = "plcs"


@dataclass(frozen=True, slots=True)
class StageDefinition:
    stage: Stage
    dependencies: tuple[Stage, ...]
    owned_path: Path
    fixed_outputs: tuple[Path, ...]


STAGES = (
    StageDefinition(
        Stage.INGEST,
        (),
        Path("source"),
        (Path("source/video.mp4"), Path("source/metadata.json")),
    ),
    StageDefinition(
        Stage.RECONSTRUCTION,
        (Stage.INGEST,),
        Path("reconstruction"),
        (
            Path("reconstruction/run.json"),
            Path("reconstruction/export/scene.json"),
        ),
    ),
    StageDefinition(
        Stage.ALIGNMENT,
        (Stage.RECONSTRUCTION,),
        Path("alignment"),
        (
            Path("alignment/ground-line-map.npz"),
            Path("alignment/ground-line-preview.png"),
            Path("alignment/court-geometry.json"),
            Path("alignment/alignment.json"),
            Path("alignment/diagnostics/fit-holdout.json"),
        ),
    ),
    StageDefinition(
        Stage.COURT_DATASET,
        (Stage.ALIGNMENT,),
        Path("datasets/court"),
        (Path("datasets/court/dataset.json"),),
    ),
    StageDefinition(
        Stage.BLCS_DATASET,
        (Stage.ALIGNMENT,),
        Path("datasets/blcs"),
        (Path("datasets/blcs/dataset.json"),),
    ),
    StageDefinition(
        Stage.PLCS_DATASET,
        (Stage.ALIGNMENT,),
        Path("datasets/plcs"),
        (Path("datasets/plcs/dataset.json"),),
    ),
    StageDefinition(
        Stage.REPORT,
        (
            Stage.COURT_DATASET,
            Stage.BLCS_DATASET,
            Stage.PLCS_DATASET,
        ),
        Path("report"),
        (Path("report/index.html"),),
    ),
)

BY_STAGE = {definition.stage: definition for definition in STAGES}
ORDER = tuple(definition.stage for definition in STAGES)
TARGET_STAGE = {
    Target.COURT: Stage.COURT_DATASET,
    Target.BLCS: Stage.BLCS_DATASET,
    Target.PLCS: Stage.PLCS_DATASET,
}


def descendants(stage: Stage, *, include_self: bool = False) -> tuple[Stage, ...]:
    """Return descendants in topological order."""
    selected = {stage} if include_self else set()
    changed = True
    while changed:
        changed = False
        for definition in STAGES:
            if definition.stage in selected:
                continue
            if stage in definition.dependencies or any(
                dependency in selected for dependency in definition.dependencies
            ):
                selected.add(definition.stage)
                changed = True
    return tuple(candidate for candidate in ORDER if candidate in selected)


def execution_order(
    from_stage: Stage, targets: tuple[Target, ...]
) -> tuple[Stage, ...]:
    """Select the exact downstream path for the requested targets."""
    wanted_datasets = {TARGET_STAGE[target] for target in targets}
    candidates = descendants(from_stage, include_self=True)
    return tuple(
        stage
        for stage in candidates
        if stage not in {Stage.COURT_DATASET, Stage.BLCS_DATASET, Stage.PLCS_DATASET}
        or stage in wanted_datasets
    )


def required_dependencies(
    stage: Stage, targets: tuple[Target, ...]
) -> tuple[Stage, ...]:
    """Resolve report's dynamic dependencies without a second graph."""
    if stage is Stage.REPORT:
        return tuple(TARGET_STAGE[target] for target in targets)
    return BY_STAGE[stage].dependencies
