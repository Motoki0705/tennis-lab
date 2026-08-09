"""Completeness and fail-closed tests for the typed stage definition graph."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import pytest

from src.synthetic_data_generation.pipeline import (
    CanonicalStageHandlers,
    DatasetTarget,
    StageExecutionSummary,
    StageInput,
    StageName,
)
from src.synthetic_data_generation.pipeline.contracts import (
    ScenePipelineRequest,
    StageExecutionContext,
    StageHandler,
)
from src.synthetic_data_generation.pipeline.registry import (
    StageRegistry,
    canonical_registry,
)


@dataclass(frozen=True)
class _Handler:
    stage: StageName

    def preflight(self, context: StageExecutionContext) -> None:
        pass

    def execute(self, context: StageExecutionContext) -> StageExecutionSummary:
        return StageExecutionSummary({"stage": self.stage.value})

    def validate(self, context: StageExecutionContext) -> None:
        pass


def _handlers() -> CanonicalStageHandlers:
    return CanonicalStageHandlers(
        ingest=_Handler(StageName.INGEST),
        reconstruction=_Handler(StageName.RECONSTRUCTION),
        alignment=_Handler(StageName.ALIGNMENT),
        court_dataset=_Handler(StageName.COURT_DATASET),
        blcs_dataset=_Handler(StageName.BLCS_DATASET),
        plcs_dataset=_Handler(StageName.PLCS_DATASET),
        report=_Handler(StageName.REPORT),
    )


def test_registry_binds_complete_lifecycle_inputs_and_derived_descendants(
    tmp_path: Path,
) -> None:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    request = ScenePipelineRequest(
        scene_id="scene-a",
        source_video=source,
        targets=frozenset({DatasetTarget.COURT}),
        from_stage=StageName.INGEST,
        config_schema="scene_pipeline_v1",
    )
    registry = canonical_registry(_handlers())

    assert tuple(item.name for item in registry.selected_for_request(request)) == (
        StageName.INGEST,
        StageName.RECONSTRUCTION,
        StageName.ALIGNMENT,
        StageName.COURT_DATASET,
        StageName.REPORT,
    )
    alignment = registry.definition(StageName.ALIGNMENT)
    assert set(alignment.descendants) == {
        StageName.COURT_DATASET,
        StageName.BLCS_DATASET,
        StageName.PLCS_DATASET,
        StageName.REPORT,
    }
    assert callable(alignment.preflight)
    assert callable(alignment.execute)
    assert callable(alignment.validate)
    assert alignment.required_inputs
    assert alignment.required_outputs
    assert alignment.summary_type is StageExecutionSummary
    assert registry.definition(StageName.PLCS_DATASET).required_outputs == (
        Path("dataset.json"),
        Path("backgrounds"),
        Path("scenes"),
        Path("diagnostics"),
    )


def test_execution_plan_rejects_cursor_outside_explicit_targets(tmp_path: Path) -> None:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    request = ScenePipelineRequest(
        scene_id="scene-a",
        source_video=source,
        targets=frozenset({DatasetTarget.COURT}),
        from_stage=StageName.PLCS_DATASET,
        config_schema="scene_pipeline_v1",
    )

    with pytest.raises(ValueError, match="not selected by request targets"):
        canonical_registry(_handlers()).execution_for_request(request)


def test_execution_plan_uses_cursor_descendants_for_execution(tmp_path: Path) -> None:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    request = ScenePipelineRequest(
        scene_id="scene-a",
        source_video=source,
        targets=frozenset(DatasetTarget),
        from_stage=StageName.COURT_DATASET,
        config_schema="scene_pipeline_v1",
    )

    plan = canonical_registry(_handlers()).execution_for_request(request)

    assert tuple(definition.name for definition in plan.retained_ancestors) == (
        StageName.INGEST,
        StageName.RECONSTRUCTION,
        StageName.ALIGNMENT,
    )
    assert tuple(definition.name for definition in plan.invalidated) == (
        StageName.COURT_DATASET,
        StageName.REPORT,
    )
    assert tuple(definition.name for definition in plan.execution) == (
        StageName.COURT_DATASET,
        StageName.REPORT,
    )


def test_execution_plan_keeps_unselected_descendants_for_stale_cleanup(
    tmp_path: Path,
) -> None:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    request = ScenePipelineRequest(
        scene_id="scene-a",
        source_video=source,
        targets=frozenset({DatasetTarget.COURT}),
        from_stage=StageName.ALIGNMENT,
        config_schema="scene_pipeline_v1",
    )

    plan = canonical_registry(_handlers()).execution_for_request(request)

    assert tuple(definition.name for definition in plan.retained_ancestors) == (
        StageName.INGEST,
        StageName.RECONSTRUCTION,
    )
    assert {definition.name for definition in plan.invalidated} == {
        StageName.ALIGNMENT,
        StageName.COURT_DATASET,
        StageName.BLCS_DATASET,
        StageName.PLCS_DATASET,
        StageName.REPORT,
    }
    assert tuple(definition.name for definition in plan.execution) == (
        StageName.ALIGNMENT,
        StageName.COURT_DATASET,
        StageName.REPORT,
    )


def test_registry_rejects_duplicate_handler_binding() -> None:
    handlers = _handlers()
    duplicated = replace(handlers, report=handlers.ingest)

    with pytest.raises(ValueError, match="multiple stage definitions"):
        canonical_registry(duplicated)


def test_registry_rejects_unknown_dependency() -> None:
    registry = canonical_registry(_handlers())
    definitions = dict(registry.definitions)
    definitions[StageName.INGEST] = replace(
        definitions[StageName.INGEST],
        dependencies=(cast(StageName, "unknown"),),
    )

    with pytest.raises(ValueError, match="Unknown dependencies"):
        StageRegistry(definitions)


def test_registry_rejects_owner_collision() -> None:
    registry = canonical_registry(_handlers())
    definitions = dict(registry.definitions)
    definitions[StageName.REPORT] = replace(
        definitions[StageName.REPORT],
        owner_relative_path=Path("datasets/court/report"),
    )

    with pytest.raises(ValueError, match="owner collision"):
        StageRegistry(definitions)


def test_registry_rejects_input_not_bound_to_ancestor_output() -> None:
    registry = canonical_registry(_handlers())
    definitions = dict(registry.definitions)
    definitions[StageName.ALIGNMENT] = replace(
        definitions[StageName.ALIGNMENT],
        required_inputs=(
            StageInput.resolved_configuration(),
            StageInput.stage_output(StageName.COURT_DATASET, "dataset.json"),
        ),
    )

    with pytest.raises(ValueError, match="not produced by an ancestor"):
        StageRegistry(definitions)


def test_stage_definition_rejects_unbound_lifecycle() -> None:
    registry = canonical_registry(_handlers())
    definition = registry.definition(StageName.INGEST)

    with pytest.raises(TypeError, match="unbound handler lifecycle"):
        replace(
            definition,
            handler=cast(StageHandler[StageExecutionSummary], object()),
        )
