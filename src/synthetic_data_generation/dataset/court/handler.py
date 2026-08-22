"""Canonical Court dataset stage handler for ``ScenePipelineRunner``."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.synthetic_data_generation.alignment.contracts import AlignmentResult
from src.synthetic_data_generation.alignment.validation import (
    validate_alignment_outputs,
)
from src.synthetic_data_generation.configuration import CourtDatasetConfiguration
from src.synthetic_data_generation.dataset.court.assembler import (
    CourtArrayValidationMode,
    assemble_court_dataset,
    validate_court_dataset,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.selection import (
    build_court_dataset_plan,
)
from src.synthetic_data_generation.dataset.court.contracts import CourtDatasetPlanAny
from src.synthetic_data_generation.dataset.court.schema import CourtDatasetSchemaVersion
from src.synthetic_data_generation.dataset.court.shards import CourtRenderResult
from src.synthetic_data_generation.dataset.runtime import PerformanceTimer
from src.synthetic_data_generation.pipeline.contracts import (
    StageExecutionContext,
    StageExecutionSummary,
    StageName,
)
from src.synthetic_data_generation.reconstruction.scene_export import (
    StandardSceneExport,
)


class CourtRenderBoundary(Protocol):
    """Injectable production/fake boundary with identical shard semantics."""

    def preflight(self, scene_path: Path) -> StandardSceneExport: ...

    def render(
        self,
        *,
        plan: CourtDatasetPlanAny,
        scene: StandardSceneExport,
        attempt_root: Path,
        attempt_token: str,
        alignment: AlignmentResult,
    ) -> CourtRenderResult: ...


@dataclass(frozen=True, slots=True)
class CourtDatasetStageHandler:
    """Plan, render, assemble, gate, diagnose, and publish one Court dataset."""

    configuration: CourtDatasetConfiguration
    profile: str
    renderer: CourtRenderBoundary

    def __post_init__(self) -> None:
        if not self.profile or self.profile != self.profile.strip():
            raise ValueError(
                "Court dataset profile must be a non-empty trimmed string."
            )

    def preflight(self, context: StageExecutionContext) -> None:
        """Validate scene export, accepted alignment, config, and renderer first."""
        scene_path, alignment_path = _upstream_paths(context)
        scene = self.renderer.preflight(scene_path)
        if scene.scene_id != context.request.scene_id:
            raise ValueError("NHT export scene_id disagrees with the pipeline request.")
        validate_alignment_outputs(alignment_path)

    def execute(self, context: StageExecutionContext) -> StageExecutionSummary:
        """Execute entirely beneath the runner-provided stage-attempt staging path."""
        timer = PerformanceTimer()
        _require_staging(context)
        scene_path, alignment_path = _upstream_paths(context)
        scene = self.renderer.preflight(scene_path)
        alignment = validate_alignment_outputs(alignment_path)
        plan = build_court_dataset_plan(
            scene_id=context.request.scene_id,
            profile=self.profile,
            cameras=scene.cameras,
            layout=alignment.layout,
            configuration=self.configuration,
            metric_adapter=alignment.metric_adapter,
        )
        attempt_root = context.staging_path / "_attempt"
        attempt_token = uuid.uuid4().hex
        rendered = self.renderer.render(
            plan=plan,
            scene=scene,
            attempt_root=attempt_root,
            attempt_token=attempt_token,
            alignment=alignment,
        )
        report = assemble_court_dataset(
            context.staging_path,
            plan=plan,
            layout=alignment.layout,
            metric_adapter=alignment.metric_adapter,
            render_result=rendered,
            configuration=self.configuration,
            attempt_root=attempt_root,
            performance_timer=timer,
        )
        values: dict[str, object] = {
            "profile": self.profile,
            "proposal_count": report.proposal_count,
            "accepted_frame_count": report.accepted_frame_count,
            "rejected_frame_count": report.rejected_frame_count,
            "accepted_fraction": report.accepted_fraction,
            "trajectory_group_count": report.trajectory_group_count,
            "maximum_adjacent_step_m": report.maximum_adjacent_step_m,
            "split_frame_counts": dict(report.split_frame_counts),
            "performance": report.performance.to_dict(),
        }
        count_key = (
            "court_group_counts"
            if self.configuration.schema_version is CourtDatasetSchemaVersion.V1
            else "court_sample_counts"
        )
        values[count_key] = dict(report.court_group_counts)
        return StageExecutionSummary(values=values)

    def validate(self, context: StageExecutionContext) -> None:
        """Revalidate the complete staged fixed inventory before publication."""
        _require_staging(context)
        if (context.staging_path / "_attempt").exists():
            raise ValueError("Attempt-local Court shards must not enter publication.")
        actual = {path.name for path in context.staging_path.iterdir()}
        expected = {"dataset.json", "samples", "diagnostics"}
        if actual != expected:
            raise ValueError(
                f"Court staging inventory mismatch; missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}."
            )
        validate_court_dataset(
            context.staging_path,
            expected_configuration=self.configuration,
            array_validation=CourtArrayValidationMode.HEADERS_ONLY,
        )


def _upstream_paths(context: StageExecutionContext) -> tuple[Path, Path]:
    if context.stage.name is not StageName.COURT_DATASET:
        raise ValueError("CourtDatasetStageHandler received a non-Court stage context.")
    owner = Path(context.owner_path)
    if owner.parts[-2:] != ("datasets", "court"):
        raise ValueError(
            "Court stage owner must be the fixed datasets/court directory."
        )
    scene_root = owner.parents[1]
    return (
        scene_root / "reconstruction" / "export" / "scene.json",
        scene_root / "alignment",
    )


def _require_staging(context: StageExecutionContext) -> None:
    if context.stage.name is not StageName.COURT_DATASET:
        raise ValueError("CourtDatasetStageHandler received a non-Court stage context.")
    expected = (
        context.owner_path.parents[1]
        / ".transactions"
        / context.stage.name.value
        / "snapshot"
    )
    if context.staging_path != expected:
        raise ValueError(
            f"Court handler requires the workspace transaction snapshot {expected}, "
            f"got {context.staging_path}."
        )
    if not context.staging_path.is_dir() or context.staging_path.is_symlink():
        raise ValueError("Court staging must be an existing ordinary directory.")


__all__ = ["CourtDatasetStageHandler", "CourtRenderBoundary"]
