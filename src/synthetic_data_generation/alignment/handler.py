"""Canonical alignment-stage handler over the public NHT scene export boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.synthetic_data_generation.alignment.contracts import (
    AlignmentAcceptancePolicy,
    AlignmentEvidence,
)
from src.synthetic_data_generation.alignment.fitting import fit_alignment
from src.synthetic_data_generation.alignment.validation import (
    validate_alignment_outputs,
    write_alignment_outputs,
)
from src.synthetic_data_generation.pipeline.contracts import (
    StageExecutionContext,
    StageExecutionSummary,
    StageName,
)
from src.synthetic_data_generation.reconstruction.scene_export import (
    StandardSceneExport,
    validate_standard_scene_export,
)


class SceneExportLoader(Protocol):
    """Callable public scene-export validator boundary."""

    def __call__(self, scene_path: str | Path) -> StandardSceneExport: ...


class AlignmentEvidenceSource(Protocol):
    """Application adapter that extracts court evidence from a validated scene export."""

    def preflight(self, scene: StandardSceneExport) -> None:
        """Validate availability/configuration without mutating alignment outputs."""

    def collect(self, scene: StandardSceneExport) -> AlignmentEvidence:
        """Return explicit fit/holdout evidence without writing stage files."""


@dataclass(frozen=True, slots=True)
class AlignmentStageHandler:
    """Fit, holdout-gate, validate, and stage one fixed alignment inventory."""

    evidence_source: AlignmentEvidenceSource
    policy: AlignmentAcceptancePolicy
    scene_loader: SceneExportLoader = validate_standard_scene_export

    def preflight(self, context: StageExecutionContext) -> None:
        """Validate the canonical NHT export before any pipeline invalidation."""
        scene_json = _scene_json_path(context)
        export_root = scene_json.parent
        required_files = (
            scene_json,
            export_root / "cameras.json",
            export_root / "points_scene.npy",
        )
        missing_files = [str(path) for path in required_files if not path.is_file()]
        required_directories = (export_root / "images", export_root / "model")
        missing_directories = [
            str(path) for path in required_directories if not path.is_dir()
        ]
        if missing_files or missing_directories:
            raise FileNotFoundError(
                "NHT standard export is incomplete; "
                f"missing_files={missing_files}, missing_directories={missing_directories}."
            )
        scene = self._load_scene(scene_json, expected_scene_id=context.request.scene_id)
        self.evidence_source.preflight(scene)
        evidence = self.evidence_source.collect(scene)
        if not isinstance(evidence, AlignmentEvidence):
            raise TypeError("Alignment evidence source returned an unsupported value.")
        fit_alignment(evidence, policy=self.policy)

    def execute(self, context: StageExecutionContext) -> StageExecutionSummary:
        """Write only to the runner-provided staging path; never use an alternate path."""
        scene_json = _scene_json_path(context)
        _require_exact_staging(context)
        scene = self._load_scene(scene_json, expected_scene_id=context.request.scene_id)
        evidence = self.evidence_source.collect(scene)
        if not isinstance(evidence, AlignmentEvidence):
            raise TypeError("Alignment evidence source returned an unsupported value.")
        result = fit_alignment(evidence, policy=self.policy)
        write_alignment_outputs(context.staging_path, evidence=evidence, result=result)
        return StageExecutionSummary(
            values={
                "evaluated_court_count": len(result.candidates),
                "accepted_court_count": len(result.layout.courts),
                "fit_camera_count": len(result.partitions.fit_camera_ids),
                "holdout_camera_count": len(result.partitions.holdout_camera_ids),
                "primary_court_instance_id": result.layout.primary_court_instance_id,
            }
        )

    def validate(self, context: StageExecutionContext) -> None:
        """Require complete semantic outputs before fixed-path publication."""
        _require_exact_staging(context)
        validate_alignment_outputs(context.staging_path)

    def _load_scene(self, path: Path, *, expected_scene_id: str) -> StandardSceneExport:
        scene = self.scene_loader(path)
        if not isinstance(scene, StandardSceneExport):
            raise TypeError("scene_loader returned an unsupported scene export value.")
        if scene.scene_id != expected_scene_id:
            raise ValueError(
                f"NHT scene_id {scene.scene_id!r} disagrees with {expected_scene_id!r}."
            )
        return scene


def _scene_json_path(context: StageExecutionContext) -> Path:
    if context.stage.name is not StageName.ALIGNMENT:
        raise ValueError(
            "AlignmentStageHandler received a non-alignment stage context."
        )
    owner_path = Path(context.owner_path)
    if owner_path.name != "alignment":
        raise ValueError("Alignment stage owner must be the fixed alignment directory.")
    return owner_path.parent / "reconstruction" / "export" / "scene.json"


def _require_exact_staging(context: StageExecutionContext) -> None:
    expected = context.owner_path / "staging"
    if context.staging_path != expected:
        raise ValueError(
            f"Alignment handler requires the provided fixed staging path {expected}, "
            f"got {context.staging_path}."
        )
    if not context.staging_path.is_dir() or context.staging_path.is_symlink():
        raise ValueError("Alignment staging must be an ordinary existing directory.")


__all__ = ["AlignmentEvidenceSource", "AlignmentStageHandler", "SceneExportLoader"]
