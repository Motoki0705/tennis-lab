"""Tests for fixed-path output, projection, binding, and stage validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from src.synthetic_data_generation.alignment.contracts import (
    AlignmentAcceptancePolicy,
    AlignmentEvidence,
    CandidateEvidence,
    CorrespondenceSet,
)
from src.synthetic_data_generation.alignment.fitting import fit_alignment
from src.synthetic_data_generation.alignment.handler import AlignmentStageHandler
from src.synthetic_data_generation.alignment.validation import (
    load_accepted_layout,
    validate_alignment_outputs,
    validate_court_transform_binding,
    validate_projection_equivalence,
    write_alignment_outputs,
)
from src.synthetic_data_generation.pipeline.contracts import (
    DatasetTarget,
    PublicationMode,
    ScenePipelineRequest,
    StageName,
    StageSpec,
)
from src.synthetic_data_generation.reconstruction.scene_export import (
    StandardSceneExport,
)
from src.synthetic_data_generation.scene_contract import RigidTransform


def test_fixed_outputs_round_trip_and_reject_cross_file_tampering(
    tmp_path: Path,
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    result = fit_alignment(alignment_evidence, policy=alignment_policy)
    staging = tmp_path / "alignment"
    staging.mkdir(parents=True)
    write_alignment_outputs(staging, evidence=alignment_evidence, result=result)

    validated = validate_alignment_outputs(staging)
    layout = load_accepted_layout(staging)
    assert validated.to_dict() == result.to_dict()
    assert layout.to_dict() == result.layout.to_dict()
    assert {path.name for path in staging.iterdir()} == {
        "ground-line-map.npz",
        "court-geometry.json",
        "alignment.json",
        "diagnostics",
    }

    geometry_path = staging / "court-geometry.json"
    original_geometry = geometry_path.read_text(encoding="utf-8")
    geometry = json.loads(original_geometry)
    geometry["fit_camera_ids"] = ["tampered"]
    geometry_path.write_text(json.dumps(geometry), encoding="utf-8")
    with pytest.raises(ValueError, match="court-geometry.json disagrees"):
        validate_alignment_outputs(staging)

    geometry_path.write_text(original_geometry, encoding="utf-8")
    archive_path = staging / "ground-line-map.npz"
    with np.load(archive_path, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    arrays["fit_points_scene"] = arrays["fit_points_scene"].copy()
    arrays["fit_points_scene"][0, 0] += 0.1
    np.savez_compressed(archive_path, **arrays)
    with pytest.raises(ValueError, match="Ground-line evidence"):
        validate_alignment_outputs(staging)


def test_ground_line_archive_rejects_wrong_dtype(
    tmp_path: Path,
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    result = fit_alignment(alignment_evidence, policy=alignment_policy)
    staging = tmp_path / "alignment" / "staging"
    staging.mkdir(parents=True)
    write_alignment_outputs(staging, evidence=alignment_evidence, result=result)
    archive_path = staging / "ground-line-map.npz"
    with np.load(archive_path, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    arrays["fit_points_scene"] = arrays["fit_points_scene"].astype(np.float32)
    np.savez_compressed(archive_path, **arrays)

    with pytest.raises(ValueError, match="dtype float64"):
        validate_alignment_outputs(staging)


def test_projection_and_target_binding_share_the_exact_court_transform(
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    result = fit_alignment(alignment_evidence, policy=alignment_policy)
    court = result.layout.courts[0]
    camera_matrix = np.eye(4, dtype=np.float64)
    camera_matrix[:3, 3] = (0.0, 0.0, -10.0)
    camera_to_court = RigidTransform.from_matrix(camera_matrix)
    points = np.asarray(
        [[-2.0, -3.0, 0.0], [2.0, -3.0, 0.0], [0.0, 3.0, 0.0]],
        dtype=np.float64,
    )
    error = validate_projection_equivalence(
        court,
        camera_to_court=camera_to_court,
        intrinsics=(1000.0, 0.0, 640.0, 0.0, 1000.0, 360.0, 0.0, 0.0, 1.0),
        points_court=points,
    )
    assert error <= 1.0e-7

    selected = validate_court_transform_binding(
        result.layout,
        court_instance_id=court.court_instance_id,
        candidate_id=court.candidate_id,
        transforms={
            "camera": court.scene_from_court,
            "ball": court.scene_from_court,
            "player": court.scene_from_court,
        },
    )
    assert selected == court

    mismatched = RigidTransform.from_matrix(
        court.scene_from_court.matrix()
        @ np.asarray(
            [
                [1.0, 0.0, 0.0, 0.1],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    )
    with pytest.raises(ValueError, match="transform mismatch"):
        validate_court_transform_binding(
            result.layout,
            court_instance_id=court.court_instance_id,
            candidate_id=court.candidate_id,
            transforms={"camera": court.scene_from_court, "trajectory": mismatched},
        )


@dataclass
class _EvidenceSource:
    evidence: AlignmentEvidence
    preflight_calls: int = 0

    def preflight(self, scene: StandardSceneExport) -> None:
        assert scene.scene_path.name == "scene.json"
        assert scene.scene_id == "scene-a"
        self.preflight_calls += 1

    def collect(self, scene: StandardSceneExport) -> AlignmentEvidence:
        assert scene.scene_path.name == "scene.json"
        assert scene.scene_id == "scene-a"
        return self.evidence


@dataclass(frozen=True)
class _Context:
    request: ScenePipelineRequest
    stage: StageSpec
    owner_path: Path
    staging_path: Path


def test_stage_handler_consumes_fixed_export_and_writes_only_to_staging(
    tmp_path: Path,
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    context = _context(tmp_path)
    source = _EvidenceSource(alignment_evidence)
    handler = AlignmentStageHandler(
        evidence_source=source,
        policy=alignment_policy,
        scene_loader=_load_test_scene,
    )

    handler.preflight(context)
    summary = handler.execute(context)
    handler.validate(context)

    assert source.preflight_calls == 1
    assert summary.values["accepted_court_count"] == 2
    assert {path.name for path in context.owner_path.iterdir()} == {"staging"}
    assert (context.staging_path / "alignment.json").is_file()

    wrong_context = _Context(
        request=context.request,
        stage=context.stage,
        owner_path=context.owner_path,
        staging_path=tmp_path / "fallback",
    )
    with pytest.raises(ValueError, match="fixed staging path"):
        handler.execute(wrong_context)


def test_stage_handler_leaves_no_outputs_when_holdout_gate_fails(
    tmp_path: Path,
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    first = alignment_evidence.candidates[0]
    rejected = CandidateEvidence(
        court_instance_id=first.court_instance_id,
        candidate_id=first.candidate_id,
        fit=first.fit,
        holdout=CorrespondenceSet(
            points_court=first.holdout.points_court,
            points_scene=first.holdout.points_scene + 2.0,
            camera_ids=first.holdout.camera_ids,
        ),
    )
    failed_evidence = AlignmentEvidence(
        partitions=alignment_evidence.partitions,
        candidates=(rejected,),
        measured_camera_lines=alignment_evidence.measured_camera_lines,
        complex_points_scene=alignment_evidence.complex_points_scene,
        primary_candidate_id=None,
        metric_adapter=alignment_evidence.metric_adapter,
        diagnostics=type(alignment_evidence.diagnostics)(
            cameras=alignment_evidence.diagnostics.cameras,
            candidate_scales=(alignment_evidence.diagnostics.candidate_scales[0],),
            common_nht_scene_units_per_metre=1.0,
            maximum_relative_scale_deviation=0.0,
        ),
    )
    context = _context(tmp_path)
    handler = AlignmentStageHandler(
        evidence_source=_EvidenceSource(failed_evidence),
        policy=alignment_policy,
        scene_loader=_load_test_scene,
    )

    with pytest.raises(ValueError, match="Holdout acceptance failed"):
        handler.preflight(context)
    assert not any(context.staging_path.iterdir())


def _context(tmp_path: Path) -> _Context:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    owner = tmp_path / "scene-a" / "alignment"
    staging = owner / "staging"
    staging.mkdir(parents=True)
    export = owner.parent / "reconstruction" / "export"
    export.mkdir(parents=True)
    (export / "scene.json").write_text("{}", encoding="utf-8")
    (export / "cameras.json").write_text("{}", encoding="utf-8")
    np.save(export / "points_scene.npy", np.zeros((1, 6), dtype=np.float32))
    (export / "images").mkdir()
    (export / "model").mkdir()
    request = ScenePipelineRequest(
        scene_id="scene-a",
        source_video=video,
        targets=frozenset({DatasetTarget.COURT}),
        from_stage=StageName.ALIGNMENT,
        config_schema="test-v1",
    )
    spec = StageSpec(
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
    )
    return _Context(
        request=request,
        stage=spec,
        owner_path=owner,
        staging_path=staging,
    )


def _load_test_scene(scene_path: str | Path) -> StandardSceneExport:
    path = Path(scene_path)
    return StandardSceneExport(
        scene_id="scene-a",
        export_root=path.parent,
        scene_path=path,
        cameras=(),
        points_scene=np.zeros((1, 6), dtype=np.float32),
        scene_from_sfm=tuple(float(value) for value in np.eye(4).ravel()),
        sfm_from_scene=tuple(float(value) for value in np.eye(4).ravel()),
        checkpoint_path=path.parent / "model" / "checkpoint.pt",
        runtime_config_path=path.parent / "model" / "config.json",
    )
