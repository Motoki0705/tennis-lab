"""Fixed-path serialization and semantic validation for alignment outputs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.alignment.contracts import (
    ALIGNMENT_COORDINATE_CONVENTION,
    AlignmentEvaluationDiagnostics,
    AlignmentEvidence,
    AlignmentEvidenceDiagnostics,
    AlignmentPartitions,
    AlignmentResult,
    AlignmentTrace,
    CameraEvidencePartition,
    CameraExclusionReason,
    CameraLineDiagnostics,
    CandidateEvidence,
    CandidateScaleDiagnostics,
    CorrespondenceSet,
    EvaluatedAlignment,
    ExcludedCameraDiagnostics,
    FixedCameraSelectionDiagnostics,
    GroundPlaneFrame,
    LineInferenceDeterminismDiagnostics,
    MeasuredCameraLines,
    MetricSceneAdapter,
    ProposalSearchDiagnostics,
)
from src.synthetic_data_generation.alignment.fitting import (
    fit_alignment,
    whole_court_diagnostics,
)
from src.synthetic_data_generation.alignment.heatmaps import (
    LINE_HEATMAP_DIRECTORY,
    AlignmentLineHeatmaps,
    validate_line_heatmaps,
    write_line_heatmaps,
)
from src.synthetic_data_generation.alignment.settings import WholeCourtEvidenceSettings
from src.synthetic_data_generation.scene_contract import (
    CourtInstance,
    MultiCourtLayout,
    RigidTransform,
)

GROUND_LINE_MAP_FILE = "ground-line-map.npz"
COURT_GEOMETRY_FILE = "court-geometry.json"
ALIGNMENT_FILE = "alignment.json"
DIAGNOSTICS_DIRECTORY = "diagnostics"

_GROUND_LINE_KEYS = {
    "schema",
    "primary_candidate_id",
    "candidate_ids",
    "court_instance_ids",
    "fit_camera_ids",
    "holdout_camera_ids",
    "fit_candidate_index",
    "fit_camera_index",
    "fit_points_court",
    "fit_points_scene",
    "holdout_candidate_index",
    "holdout_camera_index",
    "holdout_points_court",
    "holdout_points_scene",
    "complex_points_scene",
    "nht_scene_from_metric_scene",
    "metric_scene_from_nht_scene",
    "nht_scene_units_per_metre",
    "ground_plane_frame_json",
    "diagnostic_selected_line_pixel_count",
    "diagnostic_projected_line_point_count",
    "diagnostic_fixed_selection_json",
    "diagnostic_evaluation_json",
    "diagnostic_determinism_json",
    "diagnostic_proposal_search_json",
    "diagnostic_alignment_trace_json",
    "diagnostic_excluded_camera_ids",
    "diagnostic_excluded_camera_partitions",
    "diagnostic_excluded_selected_line_pixel_count",
    "diagnostic_excluded_projected_line_point_count",
    "diagnostic_excluded_camera_reasons",
    "diagnostic_candidate_scales",
    "diagnostic_template_scores",
    "diagnostic_common_scale_refit_template_scores",
    "diagnostic_common_scale_refit_candidate_centers_uv_metres",
    "diagnostic_common_scale_refit_candidate_orientations_radians",
    "diagnostic_common_scale_refit_center_displacements_metres",
    "diagnostic_maximum_common_scale_refit_center_displacements_metres",
    "diagnostic_proposal_orientation_band_minimum_radians",
    "diagnostic_proposal_orientation_band_maximum_radians",
    "diagnostic_proposal_residual_point_count_before_suppression",
    "diagnostic_proposal_residual_point_count_after_suppression",
    "diagnostic_native_candidate_centers_uv_metres",
    "diagnostic_native_candidate_orientations_radians",
    "diagnostic_maximum_relative_scale_deviation",
    "diagnostic_lattice_assisted_candidate_ids",
    "whole_court_required_court_count",
    "whole_court_maximum_common_scale_relative_deviation",
    "whole_court_maximum_center_refit_displacement_metres",
    "whole_court_minimum_distinct_offset_levels",
    "whole_court_minimum_matches_per_offset_level",
    "whole_court_minimum_level_camera_count",
    "whole_court_minimum_secondary_tangential_span_metres",
    "whole_court_minimum_longitudinal_offset_span_metres",
    "whole_court_minimum_longitudinal_tangential_span_metres",
    "whole_court_minimum_transverse_offset_span_metres",
    "whole_court_minimum_transverse_tangential_span_metres",
    "whole_court_samples_per_metre",
    "whole_court_inlier_distance_metres",
    "whole_court_minimum_inlier_fraction",
    "whole_court_maximum_q95_error_metres",
    "whole_court_minimum_semantic_segment_inlier_fraction",
    "whole_court_minimum_center_separation_metres",
    "whole_court_maximum_footprint_overlap_fraction",
    "line_camera_index",
    "line_points_nht_scene",
}
_DIAGNOSTIC_FILES = {"candidate-metrics.json", "evidence.json", "summary.txt"}


class _SavezCompressed(Protocol):
    def __call__(self, file: Path, **arrays: NDArray[Any]) -> None: ...


_savez_compressed = cast(_SavezCompressed, np.savez_compressed)


def write_alignment_outputs(
    staging_path: Path,
    *,
    evidence: AlignmentEvidence,
    result: AlignmentResult,
    heatmaps: AlignmentLineHeatmaps,
) -> None:
    """Write only the declared fixed files beneath one provided staging path."""
    _require_staging_directory(staging_path)
    if any(staging_path.iterdir()):
        raise ValueError("Alignment staging must be empty before execution.")
    if result.partitions != evidence.partitions:
        raise ValueError("Alignment result partitions disagree with source evidence.")
    candidate_ids = tuple(candidate.candidate_id for candidate in result.candidates)
    evidence_candidate_ids = tuple(
        candidate.candidate_id for candidate in evidence.candidates
    )
    if candidate_ids != evidence_candidate_ids:
        raise ValueError(
            "Alignment result candidate order disagrees with source evidence."
        )
    EvaluatedAlignment(evidence=evidence, result=result, heatmaps=heatmaps)

    archive = _evidence_archive(evidence)
    _savez_compressed(staging_path / GROUND_LINE_MAP_FILE, **archive)
    _write_json(staging_path / COURT_GEOMETRY_FILE, _court_geometry_payload(result))
    _write_json(staging_path / ALIGNMENT_FILE, result.to_dict())
    diagnostics = staging_path / DIAGNOSTICS_DIRECTORY
    diagnostics.mkdir(parents=False, exist_ok=False)
    _write_json(
        diagnostics / "candidate-metrics.json",
        _metrics_payload(result, evidence=evidence),
    )
    _write_json(diagnostics / "evidence.json", evidence.diagnostics.to_dict())
    (diagnostics / "summary.txt").write_text(_human_summary(result), encoding="utf-8")
    write_line_heatmaps(
        staging_path / LINE_HEATMAP_DIRECTORY,
        heatmaps=heatmaps,
    )


def validate_alignment_outputs(output_path: Path) -> AlignmentResult:
    """Validate the complete fixed inventory and cross-file semantic agreement."""
    _require_output_directory(output_path)
    expected = {
        GROUND_LINE_MAP_FILE,
        COURT_GEOMETRY_FILE,
        ALIGNMENT_FILE,
        DIAGNOSTICS_DIRECTORY,
        LINE_HEATMAP_DIRECTORY,
    }
    actual = {path.name for path in output_path.iterdir()}
    if actual != expected:
        raise ValueError(
            "Alignment output inventory mismatch; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}."
        )
    diagnostics = output_path / DIAGNOSTICS_DIRECTORY
    if not diagnostics.is_dir() or diagnostics.is_symlink():
        raise ValueError("Alignment diagnostics must be an ordinary directory.")
    diagnostic_actual = {path.name for path in diagnostics.iterdir()}
    if diagnostic_actual != _DIAGNOSTIC_FILES:
        raise ValueError(
            "Alignment diagnostics inventory does not match the fixed schema."
        )
    if any(not path.is_file() or path.is_symlink() for path in diagnostics.iterdir()):
        raise ValueError("Alignment diagnostics must contain ordinary files only.")

    result = load_alignment_result(output_path / ALIGNMENT_FILE)
    evidence = _load_evidence_archive(output_path / GROUND_LINE_MAP_FILE)
    try:
        recomputed = fit_alignment(evidence, policy=result.policy)
    except ValueError as error:
        raise ValueError(
            "Ground-line evidence does not produce an accepted alignment."
        ) from error
    if recomputed.to_dict() != result.to_dict():
        raise ValueError(
            "Ground-line evidence disagrees with the fitted alignment result."
        )
    heatmaps = validate_line_heatmaps(output_path / LINE_HEATMAP_DIRECTORY)
    EvaluatedAlignment(evidence=evidence, result=result, heatmaps=heatmaps)

    geometry = _load_json_object(output_path / COURT_GEOMETRY_FILE)
    if geometry != _court_geometry_payload(result):
        raise ValueError(
            "court-geometry.json disagrees with the final alignment result."
        )
    metrics = _load_json_object(diagnostics / "candidate-metrics.json")
    if metrics != _metrics_payload(result, evidence=evidence):
        raise ValueError("Alignment diagnostics disagree with the final result.")
    evidence_diagnostics = _load_json_object(diagnostics / "evidence.json")
    archived_evidence_diagnostics = evidence.diagnostics.to_dict()
    if evidence_diagnostics != archived_evidence_diagnostics:
        mismatched_keys = sorted(
            key
            for key in evidence_diagnostics.keys()
            | archived_evidence_diagnostics.keys()
            if evidence_diagnostics.get(key) != archived_evidence_diagnostics.get(key)
        )
        written_mismatches = {
            key: evidence_diagnostics.get(key) for key in mismatched_keys
        }
        archived_mismatches = {
            key: archived_evidence_diagnostics.get(key) for key in mismatched_keys
        }
        raise ValueError(
            "Measured evidence diagnostics disagree with the ground-line archive: "
            f"mismatched_keys={mismatched_keys},"
            f"written={written_mismatches},archived={archived_mismatches}."
        )
    if (diagnostics / "summary.txt").read_text(encoding="utf-8") != _human_summary(
        result
    ):
        raise ValueError("Human alignment summary disagrees with the final result.")
    return result


def load_alignment_result(path: Path) -> AlignmentResult:
    """Load exactly one canonical ``alignment.json`` and fail closed on semantics."""
    if path.name != ALIGNMENT_FILE:
        raise ValueError(f"Expected the fixed alignment file name {ALIGNMENT_FILE!r}.")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Alignment result is not an ordinary file: {path}")
    return AlignmentResult.from_dict(_load_json_object(path))


def load_alignment_evidence(path: Path) -> AlignmentEvidence:
    """Load the fixed strict archive, including required trace and plane metadata."""
    if path.name != GROUND_LINE_MAP_FILE:
        raise ValueError(
            f"Expected the fixed ground-line file name {GROUND_LINE_MAP_FILE!r}."
        )
    return _load_evidence_archive(path)


def load_accepted_layout(alignment_directory: Path) -> MultiCourtLayout:
    """Validate the full fixed inventory and return the downstream court authority."""
    if alignment_directory.name != "alignment":
        raise ValueError("Expected the fixed alignment owner directory.")
    result = validate_alignment_outputs(alignment_directory)
    if any(
        not candidate.accepted
        for candidate in result.candidates
        if candidate.court_instance_id
        in {court.court_instance_id for court in result.layout.courts}
    ):
        raise ValueError("A rejected court candidate is present in MultiCourtLayout.")
    return result.layout


def validate_projection_equivalence(
    court: CourtInstance,
    *,
    camera_to_court: RigidTransform,
    intrinsics: Sequence[float],
    points_court: NDArray[np.floating[Any]],
    atol: float = 1.0e-7,
) -> float:
    """Require court-local and transformed scene-space projections to agree."""
    intrinsic = np.asarray(intrinsics, dtype=np.float64)
    if intrinsic.size != 9:
        raise ValueError("intrinsics must contain exactly nine values.")
    intrinsic = intrinsic.reshape(3, 3)
    if (
        not np.isfinite(intrinsic).all()
        or intrinsic[0, 0] <= 0.0
        or intrinsic[1, 1] <= 0.0
    ):
        raise ValueError("intrinsics must be finite with positive focal lengths.")
    if not np.allclose(intrinsic[2], (0.0, 0.0, 1.0), atol=1.0e-9, rtol=0.0):
        raise ValueError("intrinsics must have homogeneous bottom row [0, 0, 1].")
    if not np.isfinite(atol) or atol <= 0.0:
        raise ValueError("atol must be positive and finite.")
    court_points = np.asarray(points_court, dtype=np.float64)
    if (
        court_points.ndim != 2
        or court_points.shape[1] != 3
        or not np.isfinite(court_points).all()
    ):
        raise ValueError("points_court must be a finite (N, 3) array.")
    camera_points_local = camera_to_court.inverse().apply(court_points)
    scene_from_camera = RigidTransform.from_matrix(
        court.scene_from_court.matrix() @ camera_to_court.matrix()
    )
    scene_points = court.scene_from_court.apply(court_points)
    camera_points_scene_route = scene_from_camera.inverse().apply(scene_points)
    local_pixels = _project(camera_points_local, intrinsic)
    scene_pixels = _project(camera_points_scene_route, intrinsic)
    error = (
        float(np.max(np.abs(local_pixels - scene_pixels))) if len(local_pixels) else 0.0
    )
    if error > atol:
        raise ValueError(
            f"Court-local and scene-space projection differ by {error:.3g}, above {atol:.3g}."
        )
    return error


def validate_court_transform_binding(
    layout: MultiCourtLayout,
    *,
    court_instance_id: str,
    candidate_id: str,
    transforms: Mapping[str, RigidTransform],
    atol: float = 1.0e-7,
) -> CourtInstance:
    """Reject camera/trajectory metadata that do not share one target-court transform."""
    if not transforms:
        raise ValueError("At least one named transform binding is required.")
    if any(not isinstance(name, str) or not name.strip() for name in transforms):
        raise TypeError("Transform binding names must be non-empty strings.")
    if any(
        not isinstance(transform, RigidTransform) for transform in transforms.values()
    ):
        raise TypeError("Transform bindings must contain only RigidTransform values.")
    if not np.isfinite(atol) or atol <= 0.0:
        raise ValueError("atol must be positive and finite.")
    court = layout.court(court_instance_id)
    if court.candidate_id != candidate_id:
        raise ValueError("candidate_id disagrees with the selected target court.")
    expected = court.scene_from_court.matrix()
    mismatches = [
        name
        for name, transform in transforms.items()
        if not np.allclose(transform.matrix(), expected, atol=atol, rtol=0.0)
    ]
    if mismatches:
        raise ValueError(
            f"Target-court transform mismatch for bindings: {sorted(mismatches)}."
        )
    return court


def _evidence_archive(evidence: AlignmentEvidence) -> dict[str, NDArray[Any]]:
    all_camera_ids = (
        evidence.partitions.fit_camera_ids + evidence.partitions.holdout_camera_ids
    )
    camera_indices = {
        camera_id: index for index, camera_id in enumerate(all_camera_ids)
    }
    fit_points_court: list[NDArray[np.float64]] = []
    fit_points_scene: list[NDArray[np.float64]] = []
    fit_candidate_index: list[NDArray[np.int32]] = []
    fit_camera_index: list[NDArray[np.int32]] = []
    holdout_points_court: list[NDArray[np.float64]] = []
    holdout_points_scene: list[NDArray[np.float64]] = []
    holdout_candidate_index: list[NDArray[np.int32]] = []
    holdout_camera_index: list[NDArray[np.int32]] = []
    for index, candidate in enumerate(evidence.candidates):
        fit_count = len(candidate.fit.points_court)
        holdout_count = len(candidate.holdout.points_court)
        fit_points_court.append(candidate.fit.points_court)
        fit_points_scene.append(candidate.fit.points_scene)
        fit_candidate_index.append(np.full(fit_count, index, dtype=np.int32))
        fit_camera_index.append(
            np.asarray(
                [camera_indices[item] for item in candidate.fit.camera_ids],
                dtype=np.int32,
            )
        )
        holdout_points_court.append(candidate.holdout.points_court)
        holdout_points_scene.append(candidate.holdout.points_scene)
        holdout_candidate_index.append(np.full(holdout_count, index, dtype=np.int32))
        holdout_camera_index.append(
            np.asarray(
                [camera_indices[item] for item in candidate.holdout.camera_ids],
                dtype=np.int32,
            )
        )
    line_camera_index = np.concatenate(
        [
            np.full(len(item.points_nht_scene), index, dtype=np.int32)
            for index, item in enumerate(evidence.measured_camera_lines)
        ]
    )
    line_points_nht_scene = np.concatenate(
        [item.points_nht_scene for item in evidence.measured_camera_lines]
    )
    whole_court = evidence.whole_court_settings
    return {
        "schema": np.asarray("semantic_ground_line_correspondences_v14"),
        "primary_candidate_id": np.asarray(evidence.primary_candidate_id or ""),
        "candidate_ids": np.asarray(
            [candidate.candidate_id for candidate in evidence.candidates], dtype=np.str_
        ),
        "court_instance_ids": np.asarray(
            [candidate.court_instance_id for candidate in evidence.candidates],
            dtype=np.str_,
        ),
        "fit_camera_ids": np.asarray(evidence.partitions.fit_camera_ids, dtype=np.str_),
        "holdout_camera_ids": np.asarray(
            evidence.partitions.holdout_camera_ids, dtype=np.str_
        ),
        "fit_candidate_index": np.concatenate(fit_candidate_index),
        "fit_camera_index": np.concatenate(fit_camera_index),
        "fit_points_court": np.concatenate(fit_points_court),
        "fit_points_scene": np.concatenate(fit_points_scene),
        "holdout_candidate_index": np.concatenate(holdout_candidate_index),
        "holdout_camera_index": np.concatenate(holdout_camera_index),
        "holdout_points_court": np.concatenate(holdout_points_court),
        "holdout_points_scene": np.concatenate(holdout_points_scene),
        "complex_points_scene": evidence.complex_points_scene,
        "line_camera_index": line_camera_index,
        "line_points_nht_scene": line_points_nht_scene,
        "nht_scene_from_metric_scene": evidence.metric_adapter.nht_matrix(),
        "metric_scene_from_nht_scene": evidence.metric_adapter.metric_matrix(),
        "nht_scene_units_per_metre": np.asarray(
            evidence.metric_adapter.nht_scene_units_per_metre,
            dtype=np.float64,
        ),
        "ground_plane_frame_json": np.asarray(
            json.dumps(
                evidence.ground_plane_frame.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "diagnostic_selected_line_pixel_count": np.asarray(
            [item.selected_line_pixel_count for item in evidence.diagnostics.cameras],
            dtype=np.int64,
        ),
        "diagnostic_projected_line_point_count": np.asarray(
            [item.projected_line_point_count for item in evidence.diagnostics.cameras],
            dtype=np.int64,
        ),
        "diagnostic_fixed_selection_json": np.asarray(
            json.dumps(
                evidence.diagnostics.selection.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "diagnostic_evaluation_json": np.asarray(
            json.dumps(
                evidence.diagnostics.evaluation.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "diagnostic_determinism_json": np.asarray(
            json.dumps(
                evidence.diagnostics.determinism.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "diagnostic_proposal_search_json": np.asarray(
            json.dumps(
                evidence.diagnostics.proposal_search.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "diagnostic_alignment_trace_json": np.asarray(
            json.dumps(
                evidence.alignment_trace.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "diagnostic_excluded_camera_ids": np.asarray(
            [item.camera_id for item in evidence.diagnostics.excluded_cameras],
            dtype=np.str_,
        ),
        "diagnostic_excluded_camera_partitions": np.asarray(
            [
                item.original_partition.value
                for item in evidence.diagnostics.excluded_cameras
            ],
            dtype=np.str_,
        ),
        "diagnostic_excluded_selected_line_pixel_count": np.asarray(
            [
                item.selected_line_pixel_count
                for item in evidence.diagnostics.excluded_cameras
            ],
            dtype=np.int64,
        ),
        "diagnostic_excluded_projected_line_point_count": np.asarray(
            [
                item.projected_line_point_count
                for item in evidence.diagnostics.excluded_cameras
            ],
            dtype=np.int64,
        ),
        "diagnostic_excluded_camera_reasons": np.asarray(
            [item.reason.value for item in evidence.diagnostics.excluded_cameras],
            dtype=np.str_,
        ),
        "diagnostic_candidate_scales": np.asarray(
            [
                item.nht_scene_units_per_metre
                for item in evidence.diagnostics.candidate_scales
            ],
            dtype=np.float64,
        ),
        "diagnostic_template_scores": np.asarray(
            [item.template_score for item in evidence.diagnostics.candidate_scales],
            dtype=np.float64,
        ),
        "diagnostic_common_scale_refit_template_scores": np.asarray(
            [
                item.common_scale_refit_template_score
                for item in evidence.diagnostics.candidate_scales
            ],
            dtype=np.float64,
        ),
        "diagnostic_common_scale_refit_candidate_centers_uv_metres": np.asarray(
            [
                item.common_scale_refit_center_uv_metres
                for item in evidence.diagnostics.candidate_scales
            ],
            dtype=np.float64,
        ),
        "diagnostic_common_scale_refit_candidate_orientations_radians": np.asarray(
            [
                item.common_scale_refit_orientation_radians
                for item in evidence.diagnostics.candidate_scales
            ],
            dtype=np.float64,
        ),
        "diagnostic_common_scale_refit_center_displacements_metres": np.asarray(
            [
                item.common_scale_refit_center_displacement_metres
                for item in evidence.diagnostics.candidate_scales
            ],
            dtype=np.float64,
        ),
        "diagnostic_maximum_common_scale_refit_center_displacements_metres": (
            np.asarray(
                [
                    item.maximum_common_scale_refit_center_displacement_metres
                    for item in evidence.diagnostics.candidate_scales
                ],
                dtype=np.float64,
            )
        ),
        "diagnostic_proposal_orientation_band_minimum_radians": np.asarray(
            [
                item.proposal_orientation_band_minimum_radians
                for item in evidence.diagnostics.candidate_scales
            ],
            dtype=np.float64,
        ),
        "diagnostic_proposal_orientation_band_maximum_radians": np.asarray(
            [
                item.proposal_orientation_band_maximum_radians
                for item in evidence.diagnostics.candidate_scales
            ],
            dtype=np.float64,
        ),
        "diagnostic_proposal_residual_point_count_before_suppression": np.asarray(
            [
                item.proposal_residual_point_count_before_suppression
                for item in evidence.diagnostics.candidate_scales
            ],
            dtype=np.int64,
        ),
        "diagnostic_proposal_residual_point_count_after_suppression": np.asarray(
            [
                item.proposal_residual_point_count_after_suppression
                for item in evidence.diagnostics.candidate_scales
            ],
            dtype=np.int64,
        ),
        "diagnostic_native_candidate_centers_uv_metres": np.asarray(
            [
                item.native_center_uv_metres
                for item in evidence.diagnostics.candidate_scales
            ],
            dtype=np.float64,
        ),
        "diagnostic_native_candidate_orientations_radians": np.asarray(
            [
                item.native_orientation_radians
                for item in evidence.diagnostics.candidate_scales
            ],
            dtype=np.float64,
        ),
        "diagnostic_maximum_relative_scale_deviation": np.asarray(
            evidence.diagnostics.maximum_relative_scale_deviation,
            dtype=np.float64,
        ),
        "diagnostic_lattice_assisted_candidate_ids": np.asarray(
            evidence.diagnostics.lattice_assisted_candidate_ids,
            dtype=np.str_,
        ),
        "whole_court_required_court_count": np.asarray(
            whole_court.required_court_count,
            dtype=np.int64,
        ),
        "whole_court_maximum_common_scale_relative_deviation": (
            _required_policy_float(
                whole_court,
                "maximum_common_scale_relative_deviation",
            )
        ),
        "whole_court_maximum_center_refit_displacement_metres": (
            _required_policy_float(
                whole_court,
                "maximum_center_refit_displacement_metres",
            )
        ),
        "whole_court_minimum_distinct_offset_levels": np.asarray(
            whole_court.minimum_distinct_offset_levels,
            dtype=np.int64,
        ),
        "whole_court_minimum_matches_per_offset_level": np.asarray(
            whole_court.minimum_matches_per_offset_level,
            dtype=np.int64,
        ),
        "whole_court_minimum_level_camera_count": np.asarray(
            whole_court.minimum_level_camera_count,
            dtype=np.int64,
        ),
        "whole_court_minimum_secondary_tangential_span_metres": (
            _required_policy_float(
                whole_court,
                "minimum_secondary_tangential_span_metres",
            )
        ),
        "whole_court_minimum_longitudinal_offset_span_metres": (
            _required_policy_float(
                whole_court,
                "minimum_longitudinal_offset_span_metres",
            )
        ),
        "whole_court_minimum_longitudinal_tangential_span_metres": (
            _required_policy_float(
                whole_court,
                "minimum_longitudinal_tangential_span_metres",
            )
        ),
        "whole_court_minimum_transverse_offset_span_metres": (
            _required_policy_float(
                whole_court,
                "minimum_transverse_offset_span_metres",
            )
        ),
        "whole_court_minimum_transverse_tangential_span_metres": (
            _required_policy_float(
                whole_court,
                "minimum_transverse_tangential_span_metres",
            )
        ),
        "whole_court_samples_per_metre": _required_policy_float(
            whole_court, "samples_per_metre"
        ),
        "whole_court_inlier_distance_metres": _required_policy_float(
            whole_court, "inlier_distance_metres"
        ),
        "whole_court_minimum_inlier_fraction": _required_policy_float(
            whole_court, "minimum_inlier_fraction"
        ),
        "whole_court_maximum_q95_error_metres": _required_policy_float(
            whole_court, "maximum_q95_error_metres"
        ),
        "whole_court_minimum_semantic_segment_inlier_fraction": (
            _required_policy_float(
                whole_court,
                "minimum_semantic_segment_inlier_fraction",
            )
        ),
        "whole_court_minimum_center_separation_metres": _required_policy_float(
            whole_court, "minimum_center_separation_metres"
        ),
        "whole_court_maximum_footprint_overlap_fraction": _required_policy_float(
            whole_court, "maximum_footprint_overlap_fraction"
        ),
    }


def _load_evidence_archive(path: Path) -> AlignmentEvidence:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Ground-line evidence is not an ordinary file: {path}")
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != _GROUND_LINE_KEYS:
            raise ValueError("Ground-line archive keys do not match the strict schema.")
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    schema = arrays["schema"]
    if (
        schema.ndim != 0
        or schema.dtype.kind != "U"
        or str(schema.item()) != "semantic_ground_line_correspondences_v14"
    ):
        raise ValueError("Unsupported ground-line correspondence schema.")
    primary = arrays["primary_candidate_id"]
    if primary.ndim != 0 or primary.dtype.kind != "U":
        raise ValueError("primary_candidate_id must be a Unicode scalar.")
    primary_candidate_id = str(primary.item()) or None
    string_names = (
        "candidate_ids",
        "court_instance_ids",
        "fit_camera_ids",
        "holdout_camera_ids",
    )
    for name in string_names:
        if (
            arrays[name].ndim != 1
            or arrays[name].dtype.kind != "U"
            or len(arrays[name]) == 0
        ):
            raise ValueError(f"Ground-line {name} must be a non-empty Unicode vector.")
        values = arrays[name].tolist()
        if len(values) != len(set(values)):
            raise ValueError(f"Ground-line {name} values must be unique.")
    if set(arrays["fit_camera_ids"].tolist()).intersection(
        arrays["holdout_camera_ids"].tolist()
    ):
        raise ValueError("Ground-line fit and holdout camera IDs overlap.")
    if len(arrays["candidate_ids"]) != len(arrays["court_instance_ids"]):
        raise ValueError("Ground-line candidate and court ID counts differ.")
    retained_camera_ids = set(arrays["fit_camera_ids"].tolist()) | set(
        arrays["holdout_camera_ids"].tolist()
    )
    _validate_excluded_camera_arrays(
        arrays,
        retained_camera_ids=retained_camera_ids,
    )
    all_camera_count = len(arrays["fit_camera_ids"]) + len(arrays["holdout_camera_ids"])
    candidate_count = len(arrays["candidate_ids"])
    _validate_adapter_and_diagnostics_arrays(
        arrays,
        camera_count=all_camera_count,
        candidate_count=candidate_count,
    )
    _validate_measured_line_arrays(arrays, camera_count=all_camera_count)
    _validate_archive_partition(
        arrays,
        prefix="fit",
        candidate_count=candidate_count,
        camera_count=all_camera_count,
        allowed_camera_indices=set(range(len(arrays["fit_camera_ids"]))),
    )
    _validate_archive_partition(
        arrays,
        prefix="holdout",
        candidate_count=candidate_count,
        camera_count=all_camera_count,
        allowed_camera_indices=set(
            range(len(arrays["fit_camera_ids"]), all_camera_count)
        ),
    )
    complex_points = arrays["complex_points_scene"]
    if (
        complex_points.dtype != np.float64
        or complex_points.ndim != 2
        or complex_points.shape[1] != 3
    ):
        raise ValueError("complex_points_scene must be a float64 (N, 3) array.")
    if (
        len(complex_points) < 2
        or not np.isfinite(complex_points).all()
        or np.any(np.ptp(complex_points, axis=0) <= 0.0)
    ):
        raise ValueError("complex_points_scene must define finite positive 3-D bounds.")
    candidate_ids = cast(list[str], arrays["candidate_ids"].tolist())
    court_instance_ids = cast(list[str], arrays["court_instance_ids"].tolist())
    fit_camera_ids = cast(list[str], arrays["fit_camera_ids"].tolist())
    holdout_camera_ids = cast(list[str], arrays["holdout_camera_ids"].tolist())
    all_camera_ids = fit_camera_ids + holdout_camera_ids
    candidates: list[CandidateEvidence] = []
    for index, (court_instance_id, candidate_id) in enumerate(
        zip(court_instance_ids, candidate_ids, strict=True)
    ):
        fit_mask = arrays["fit_candidate_index"] == index
        holdout_mask = arrays["holdout_candidate_index"] == index
        candidates.append(
            CandidateEvidence(
                court_instance_id=court_instance_id,
                candidate_id=candidate_id,
                fit=CorrespondenceSet(
                    points_court=arrays["fit_points_court"][fit_mask],
                    points_scene=arrays["fit_points_scene"][fit_mask],
                    camera_ids=tuple(
                        all_camera_ids[int(item)]
                        for item in arrays["fit_camera_index"][fit_mask]
                    ),
                ),
                holdout=CorrespondenceSet(
                    points_court=arrays["holdout_points_court"][holdout_mask],
                    points_scene=arrays["holdout_points_scene"][holdout_mask],
                    camera_ids=tuple(
                        all_camera_ids[int(item)]
                        for item in arrays["holdout_camera_index"][holdout_mask]
                    ),
                ),
            )
        )
    adapter = MetricSceneAdapter(
        nht_scene_from_metric_scene=tuple(
            float(value) for value in arrays["nht_scene_from_metric_scene"].ravel()
        ),
        metric_scene_from_nht_scene=tuple(
            float(value) for value in arrays["metric_scene_from_nht_scene"].ravel()
        ),
        nht_scene_units_per_metre=float(arrays["nht_scene_units_per_metre"].item()),
    )
    ground_plane_frame = GroundPlaneFrame.from_dict(
        _load_json_scalar(arrays, "ground_plane_frame_json")
    )
    alignment_trace = AlignmentTrace.from_dict(
        _load_json_scalar(arrays, "diagnostic_alignment_trace_json")
    )
    excluded_cameras = tuple(
        ExcludedCameraDiagnostics(
            camera_id=str(camera_id),
            original_partition=CameraEvidencePartition(
                str(arrays["diagnostic_excluded_camera_partitions"][index])
            ),
            selected_line_pixel_count=int(
                arrays["diagnostic_excluded_selected_line_pixel_count"][index]
            ),
            projected_line_point_count=int(
                arrays["diagnostic_excluded_projected_line_point_count"][index]
            ),
            reason=CameraExclusionReason(
                str(arrays["diagnostic_excluded_camera_reasons"][index])
            ),
        )
        for index, camera_id in enumerate(
            arrays["diagnostic_excluded_camera_ids"].tolist()
        )
    )
    diagnostics = AlignmentEvidenceDiagnostics(
        cameras=tuple(
            CameraLineDiagnostics(
                camera_id=camera_id,
                selected_line_pixel_count=int(
                    arrays["diagnostic_selected_line_pixel_count"][index]
                ),
                projected_line_point_count=int(
                    arrays["diagnostic_projected_line_point_count"][index]
                ),
            )
            for index, camera_id in enumerate(all_camera_ids)
        ),
        candidate_scales=tuple(
            CandidateScaleDiagnostics(
                candidate_id=candidate_id,
                nht_scene_units_per_metre=float(
                    arrays["diagnostic_candidate_scales"][index]
                ),
                template_score=float(arrays["diagnostic_template_scores"][index]),
                common_scale_refit_template_score=float(
                    arrays["diagnostic_common_scale_refit_template_scores"][index]
                ),
                common_scale_refit_center_uv_metres=(
                    float(
                        arrays[
                            "diagnostic_common_scale_refit_candidate_centers_uv_metres"
                        ][index, 0]
                    ),
                    float(
                        arrays[
                            "diagnostic_common_scale_refit_candidate_centers_uv_metres"
                        ][index, 1]
                    ),
                ),
                common_scale_refit_orientation_radians=float(
                    arrays[
                        "diagnostic_common_scale_refit_candidate_orientations_radians"
                    ][index]
                ),
                common_scale_refit_center_displacement_metres=float(
                    arrays["diagnostic_common_scale_refit_center_displacements_metres"][
                        index
                    ]
                ),
                maximum_common_scale_refit_center_displacement_metres=float(
                    arrays[
                        "diagnostic_maximum_common_scale_refit_center_displacements_metres"
                    ][index]
                ),
                proposal_orientation_band_minimum_radians=float(
                    arrays["diagnostic_proposal_orientation_band_minimum_radians"][
                        index
                    ]
                ),
                proposal_orientation_band_maximum_radians=float(
                    arrays["diagnostic_proposal_orientation_band_maximum_radians"][
                        index
                    ]
                ),
                proposal_residual_point_count_before_suppression=int(
                    arrays[
                        "diagnostic_proposal_residual_point_count_before_suppression"
                    ][index]
                ),
                proposal_residual_point_count_after_suppression=int(
                    arrays[
                        "diagnostic_proposal_residual_point_count_after_suppression"
                    ][index]
                ),
                native_center_uv_metres=(
                    float(
                        arrays["diagnostic_native_candidate_centers_uv_metres"][
                            index, 0
                        ]
                    ),
                    float(
                        arrays["diagnostic_native_candidate_centers_uv_metres"][
                            index, 1
                        ]
                    ),
                ),
                native_orientation_radians=float(
                    arrays["diagnostic_native_candidate_orientations_radians"][index]
                ),
            )
            for index, candidate_id in enumerate(candidate_ids)
        ),
        common_nht_scene_units_per_metre=adapter.nht_scene_units_per_metre,
        maximum_relative_scale_deviation=float(
            arrays["diagnostic_maximum_relative_scale_deviation"].item()
        ),
        selection=FixedCameraSelectionDiagnostics.from_dict(
            _load_json_scalar(arrays, "diagnostic_fixed_selection_json")
        ),
        evaluation=AlignmentEvaluationDiagnostics.from_dict(
            _load_json_scalar(arrays, "diagnostic_evaluation_json")
        ),
        determinism=LineInferenceDeterminismDiagnostics.from_dict(
            _load_json_scalar(arrays, "diagnostic_determinism_json")
        ),
        proposal_search=ProposalSearchDiagnostics.from_dict(
            _load_json_scalar(arrays, "diagnostic_proposal_search_json")
        ),
        excluded_cameras=excluded_cameras,
        ground_plane_frame=ground_plane_frame,
        alignment_trace=alignment_trace,
        lattice_assisted_candidate_ids=tuple(
            str(item)
            for item in arrays["diagnostic_lattice_assisted_candidate_ids"].tolist()
        ),
    )
    whole_court_settings = _load_whole_court_settings(arrays)
    return AlignmentEvidence(
        partitions=AlignmentPartitions(
            fit_camera_ids=tuple(fit_camera_ids),
            holdout_camera_ids=tuple(holdout_camera_ids),
        ),
        candidates=tuple(candidates),
        measured_camera_lines=tuple(
            MeasuredCameraLines(
                camera_id=camera_id,
                points_nht_scene=arrays["line_points_nht_scene"][
                    arrays["line_camera_index"] == index
                ],
            )
            for index, camera_id in enumerate(all_camera_ids)
        ),
        complex_points_scene=arrays["complex_points_scene"],
        primary_candidate_id=primary_candidate_id,
        metric_adapter=adapter,
        diagnostics=diagnostics,
        whole_court_settings=whole_court_settings,
    )


def _validate_measured_line_arrays(
    arrays: Mapping[str, NDArray[Any]],
    *,
    camera_count: int,
) -> None:
    points = arrays["line_points_nht_scene"]
    camera_index = arrays["line_camera_index"]
    if (
        points.dtype != np.float64
        or points.ndim != 2
        or points.shape[1] != 3
        or not np.isfinite(points).all()
    ):
        raise ValueError(
            "Measured NHT line points must be a finite float64 (N, 3) array."
        )
    if (
        camera_index.dtype != np.int32
        or camera_index.shape != (len(points),)
        or np.any(camera_index < 0)
        or np.any(camera_index >= camera_count)
    ):
        raise ValueError("Measured line camera indices are invalid.")
    if not np.array_equal(
        np.unique(camera_index),
        np.arange(camera_count, dtype=np.int32),
    ):
        raise ValueError("Every declared camera must retain measured line points.")


def _validate_excluded_camera_arrays(
    arrays: Mapping[str, NDArray[Any]],
    *,
    retained_camera_ids: set[str],
) -> None:
    ids = arrays["diagnostic_excluded_camera_ids"]
    partitions = arrays["diagnostic_excluded_camera_partitions"]
    reasons = arrays["diagnostic_excluded_camera_reasons"]
    selected_counts = arrays["diagnostic_excluded_selected_line_pixel_count"]
    projected_counts = arrays["diagnostic_excluded_projected_line_point_count"]
    if any(
        array.ndim != 1 or array.dtype.kind != "U"
        for array in (ids, partitions, reasons)
    ):
        raise ValueError("Excluded camera identifiers must be Unicode vectors.")
    count = len(ids)
    if partitions.shape != (count,) or reasons.shape != (count,):
        raise ValueError("Excluded camera diagnostic vector shapes disagree.")
    if (
        selected_counts.dtype != np.int64
        or projected_counts.dtype != np.int64
        or selected_counts.shape != (count,)
        or projected_counts.shape != (count,)
        or np.any(selected_counts < 0)
        or np.any(projected_counts < 0)
        or np.any(projected_counts > selected_counts)
    ):
        raise ValueError("Excluded camera counts must be consistent int64 vectors.")
    excluded_ids = [str(item) for item in ids.tolist()]
    if any(not item for item in excluded_ids) or len(excluded_ids) != len(
        set(excluded_ids)
    ):
        raise ValueError("Excluded camera IDs must be non-empty and unique.")
    overlap = retained_camera_ids.intersection(excluded_ids)
    if overlap:
        raise ValueError(
            f"Retained and excluded archive camera IDs overlap: {sorted(overlap)}."
        )
    allowed_partitions = {item.value for item in CameraEvidencePartition}
    if set(str(item) for item in partitions.tolist()) - allowed_partitions:
        raise ValueError("Excluded camera partitions are invalid.")
    allowed_reasons = {item.value for item in CameraExclusionReason}
    if set(str(item) for item in reasons.tolist()) - allowed_reasons:
        raise ValueError("Excluded camera reasons are invalid.")


def _load_json_scalar(
    arrays: Mapping[str, NDArray[Any]],
    name: str,
) -> object:
    encoded = arrays[name]
    if encoded.ndim != 0 or encoded.dtype.kind != "U":
        raise ValueError(f"{name} must be one Unicode JSON scalar.")
    try:
        return json.loads(str(encoded.item()))
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not valid JSON.") from error


def _validate_adapter_and_diagnostics_arrays(
    arrays: Mapping[str, NDArray[Any]],
    *,
    camera_count: int,
    candidate_count: int,
) -> None:
    for name in ("nht_scene_from_metric_scene", "metric_scene_from_nht_scene"):
        array = arrays[name]
        if (
            array.dtype != np.float64
            or array.shape != (4, 4)
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"{name} must be a finite float64 (4, 4) array.")
    scalar_names = (
        "nht_scene_units_per_metre",
        "diagnostic_maximum_relative_scale_deviation",
    )
    for name in scalar_names:
        array = arrays[name]
        if (
            array.dtype != np.float64
            or array.ndim != 0
            or not np.isfinite(array.item())
        ):
            raise ValueError(f"{name} must be a finite float64 scalar.")
    camera_names = (
        "diagnostic_selected_line_pixel_count",
        "diagnostic_projected_line_point_count",
    )
    for name in camera_names:
        array = arrays[name]
        if (
            array.dtype != np.int64
            or array.shape != (camera_count,)
            or np.any(array < 1)
        ):
            raise ValueError(
                f"{name} must contain one positive int64 value per camera."
            )
    candidate_names = (
        "diagnostic_candidate_scales",
        "diagnostic_template_scores",
        "diagnostic_common_scale_refit_template_scores",
    )
    for name in candidate_names:
        array = arrays[name]
        if (
            array.dtype != np.float64
            or array.shape != (candidate_count,)
            or not np.isfinite(array).all()
            or np.any(array <= 0.0)
        ):
            raise ValueError(
                f"{name} must contain one positive float64 value per candidate."
            )
    refit_names = (
        "diagnostic_common_scale_refit_center_displacements_metres",
        "diagnostic_maximum_common_scale_refit_center_displacements_metres",
    )
    for name in refit_names:
        array = arrays[name]
        if (
            array.dtype != np.float64
            or array.shape != (candidate_count,)
            or not np.isfinite(array).all()
            or np.any(array < 0.0)
        ):
            raise ValueError(
                f"{name} must contain one non-negative float64 per candidate."
            )
    if np.any(
        arrays["diagnostic_common_scale_refit_center_displacements_metres"]
        > arrays["diagnostic_maximum_common_scale_refit_center_displacements_metres"]
        + 1.0e-10
    ):
        raise ValueError("A common-scale refit displacement exceeds its maximum.")
    band_minimum = arrays["diagnostic_proposal_orientation_band_minimum_radians"]
    band_maximum = arrays["diagnostic_proposal_orientation_band_maximum_radians"]
    if (
        band_minimum.dtype != np.float64
        or band_maximum.dtype != np.float64
        or band_minimum.shape != (candidate_count,)
        or band_maximum.shape != (candidate_count,)
        or not np.isfinite(band_minimum).all()
        or not np.isfinite(band_maximum).all()
        or np.any(band_minimum >= band_maximum)
        or np.any(band_maximum - band_minimum > math.pi / 2.0 + 1.0e-12)
    ):
        raise ValueError("Proposal orientation-band diagnostics are invalid.")
    residual_before = arrays[
        "diagnostic_proposal_residual_point_count_before_suppression"
    ]
    residual_after = arrays[
        "diagnostic_proposal_residual_point_count_after_suppression"
    ]
    if (
        residual_before.dtype != np.int64
        or residual_after.dtype != np.int64
        or residual_before.shape != (candidate_count,)
        or residual_after.shape != (candidate_count,)
        or np.any(residual_before < 3)
        or np.any(residual_after < 0)
        or np.any(residual_after >= residual_before)
    ):
        raise ValueError("Proposal residual-count diagnostics are invalid.")
    native_centers = arrays["diagnostic_native_candidate_centers_uv_metres"]
    native_orientations = arrays["diagnostic_native_candidate_orientations_radians"]
    if (
        native_centers.dtype != np.float64
        or native_centers.shape != (candidate_count, 2)
        or native_orientations.dtype != np.float64
        or native_orientations.shape != (candidate_count,)
        or not np.isfinite(native_centers).all()
        or not np.isfinite(native_orientations).all()
        or np.any(native_orientations < band_minimum - 1.0e-12)
        or np.any(native_orientations > band_maximum + 1.0e-12)
    ):
        raise ValueError("Native proposal pose diagnostics are invalid.")
    refit_centers = arrays[
        "diagnostic_common_scale_refit_candidate_centers_uv_metres"
    ]
    refit_orientations = arrays[
        "diagnostic_common_scale_refit_candidate_orientations_radians"
    ]
    if (
        refit_centers.dtype != np.float64
        or refit_centers.shape != (candidate_count, 2)
        or refit_orientations.dtype != np.float64
        or refit_orientations.shape != (candidate_count,)
        or not np.isfinite(refit_centers).all()
        or not np.isfinite(refit_orientations).all()
    ):
        raise ValueError("Common-scale refit pose diagnostics are invalid.")
    policy_integer_names = (
        "whole_court_required_court_count",
        "whole_court_minimum_distinct_offset_levels",
        "whole_court_minimum_matches_per_offset_level",
        "whole_court_minimum_level_camera_count",
    )
    policy_integers: list[int] = []
    for name in policy_integer_names:
        array = arrays[name]
        if array.dtype != np.int64 or array.ndim != 0 or int(array.item()) < 1:
            raise ValueError(f"{name} must be a positive int64 scalar.")
        policy_integers.append(int(array.item()))
    policy_names = (
        "whole_court_maximum_common_scale_relative_deviation",
        "whole_court_maximum_center_refit_displacement_metres",
        "whole_court_minimum_longitudinal_offset_span_metres",
        "whole_court_minimum_longitudinal_tangential_span_metres",
        "whole_court_minimum_transverse_offset_span_metres",
        "whole_court_minimum_transverse_tangential_span_metres",
        "whole_court_minimum_secondary_tangential_span_metres",
        "whole_court_samples_per_metre",
        "whole_court_inlier_distance_metres",
        "whole_court_minimum_inlier_fraction",
        "whole_court_maximum_q95_error_metres",
        "whole_court_minimum_semantic_segment_inlier_fraction",
        "whole_court_minimum_center_separation_metres",
        "whole_court_maximum_footprint_overlap_fraction",
    )
    policy_values = []
    for name in policy_names:
        array = arrays[name]
        if array.dtype != np.float64 or array.ndim != 0:
            raise ValueError(f"{name} must be a float64 scalar.")
        policy_values.append(float(array.item()))
    if not all(np.isfinite(value) for value in policy_values):
        raise ValueError("Current-schema whole-court policy fields must all be finite.")


def _validate_archive_partition(
    arrays: Mapping[str, NDArray[Any]],
    *,
    prefix: str,
    candidate_count: int,
    camera_count: int,
    allowed_camera_indices: set[int],
) -> None:
    points_court = arrays[f"{prefix}_points_court"]
    points_scene = arrays[f"{prefix}_points_scene"]
    candidate_index = arrays[f"{prefix}_candidate_index"]
    camera_index = arrays[f"{prefix}_camera_index"]
    if points_court.dtype != np.float64 or points_scene.dtype != np.float64:
        raise ValueError(f"{prefix} correspondence points must have dtype float64.")
    if (
        points_court.ndim != 2
        or points_court.shape[1] != 3
        or points_scene.shape != points_court.shape
        or not np.isfinite(points_court).all()
        or not np.isfinite(points_scene).all()
    ):
        raise ValueError(f"{prefix} correspondence point arrays are invalid.")
    if candidate_index.dtype != np.int32 or camera_index.dtype != np.int32:
        raise ValueError(f"{prefix} index arrays must have dtype int32.")
    if candidate_index.shape != (len(points_court),) or camera_index.shape != (
        len(points_court),
    ):
        raise ValueError(f"{prefix} correspondence index shapes are invalid.")
    if np.any(candidate_index < 0) or np.any(candidate_index >= candidate_count):
        raise ValueError(f"{prefix} candidate indices are out of range.")
    if np.any(camera_index < 0) or np.any(camera_index >= camera_count):
        raise ValueError(f"{prefix} camera indices are out of range.")
    if not set(int(value) for value in camera_index).issubset(allowed_camera_indices):
        raise ValueError(f"{prefix} observations reference the wrong camera partition.")
    counts = np.bincount(candidate_index, minlength=candidate_count)
    if np.any(counts < 3):
        raise ValueError(
            f"Every candidate requires at least three {prefix} correspondences."
        )
    for index in range(candidate_count):
        observed = set(int(value) for value in camera_index[candidate_index == index])
        if not observed or not observed.issubset(allowed_camera_indices):
            raise ValueError(
                f"Every candidate must use measured cameras from the declared {prefix} "
                "partition."
            )


def _court_geometry_payload(result: AlignmentResult) -> dict[str, object]:
    return {
        "schema": "fitted_court_geometry_v2",
        "coordinate_convention": ALIGNMENT_COORDINATE_CONVENTION,
        "metric_scene_adapter": result.metric_adapter.to_dict(),
        "fit_camera_ids": list(result.partitions.fit_camera_ids),
        "candidates": [
            {
                "court_instance_id": candidate.court_instance_id,
                "candidate_id": candidate.candidate_id,
                "scene_from_court": candidate.scene_from_court.to_list(),
                "court_from_scene": candidate.court_from_scene.to_list(),
                "fit": candidate.fit.to_dict(),
            }
            for candidate in result.candidates
        ],
    }


def _metrics_payload(
    result: AlignmentResult,
    *,
    evidence: AlignmentEvidence,
) -> dict[str, object]:
    return {
        "schema": "alignment_candidate_metrics_v6",
        "accepted_court_instance_ids": [
            court.court_instance_id for court in result.layout.courts
        ],
        "candidates": [
            {
                "court_instance_id": candidate.court_instance_id,
                "candidate_id": candidate.candidate_id,
                "fit": candidate.fit.to_dict(),
                "holdout": candidate.holdout.to_dict(),
                "accepted": candidate.accepted,
            }
            for candidate in result.candidates
        ],
        "whole_court": whole_court_diagnostics(
            evidence,
            candidates=result.candidates,
            policy=result.policy,
        ),
    }


def _required_policy_float(
    settings: WholeCourtEvidenceSettings,
    name: str,
) -> NDArray[np.float64]:
    return np.asarray(float(getattr(settings, name)), dtype=np.float64)


def _load_whole_court_settings(
    arrays: Mapping[str, NDArray[Any]],
) -> WholeCourtEvidenceSettings:
    required = int(arrays["whole_court_required_court_count"].item())
    return WholeCourtEvidenceSettings(
        required_court_count=required,
        maximum_common_scale_relative_deviation=float(
            arrays["whole_court_maximum_common_scale_relative_deviation"].item()
        ),
        maximum_center_refit_displacement_metres=float(
            arrays["whole_court_maximum_center_refit_displacement_metres"].item()
        ),
        minimum_distinct_offset_levels=int(
            arrays["whole_court_minimum_distinct_offset_levels"].item()
        ),
        minimum_matches_per_offset_level=int(
            arrays["whole_court_minimum_matches_per_offset_level"].item()
        ),
        minimum_level_camera_count=int(
            arrays["whole_court_minimum_level_camera_count"].item()
        ),
        minimum_secondary_tangential_span_metres=float(
            arrays["whole_court_minimum_secondary_tangential_span_metres"].item()
        ),
        minimum_longitudinal_offset_span_metres=float(
            arrays["whole_court_minimum_longitudinal_offset_span_metres"].item()
        ),
        minimum_longitudinal_tangential_span_metres=float(
            arrays["whole_court_minimum_longitudinal_tangential_span_metres"].item()
        ),
        minimum_transverse_offset_span_metres=float(
            arrays["whole_court_minimum_transverse_offset_span_metres"].item()
        ),
        minimum_transverse_tangential_span_metres=float(
            arrays["whole_court_minimum_transverse_tangential_span_metres"].item()
        ),
        samples_per_metre=float(arrays["whole_court_samples_per_metre"].item()),
        inlier_distance_metres=float(
            arrays["whole_court_inlier_distance_metres"].item()
        ),
        minimum_inlier_fraction=float(
            arrays["whole_court_minimum_inlier_fraction"].item()
        ),
        maximum_q95_error_metres=float(
            arrays["whole_court_maximum_q95_error_metres"].item()
        ),
        minimum_semantic_segment_inlier_fraction=float(
            arrays["whole_court_minimum_semantic_segment_inlier_fraction"].item()
        ),
        minimum_center_separation_metres=float(
            arrays["whole_court_minimum_center_separation_metres"].item()
        ),
        maximum_footprint_overlap_fraction=float(
            arrays["whole_court_maximum_footprint_overlap_fraction"].item()
        ),
    )


def _human_summary(result: AlignmentResult) -> str:
    lines = [
        "Semantic multi-court alignment",
        f"fit cameras: {len(result.partitions.fit_camera_ids)}",
        f"holdout cameras: {len(result.partitions.holdout_camera_ids)}",
        f"evaluated courts: {len(result.candidates)}",
        f"accepted courts: {len(result.layout.courts)}",
        f"NHT scene units per metre: {result.metric_adapter.nht_scene_units_per_metre:.9g}",
    ]
    lines.extend(
        f"{candidate.court_instance_id} ({candidate.candidate_id}): "
        f"fit={candidate.fit.status.value}, holdout={candidate.holdout.status.value}"
        for candidate in result.candidates
    )
    return "\n".join(lines) + "\n"


def _project(
    points_camera: NDArray[np.float64], intrinsic: NDArray[np.float64]
) -> NDArray[np.float64]:
    if np.any(points_camera[:, 2] <= 0.0):
        raise ValueError(
            "Projection equivalence points must have positive camera-Z depth."
        )
    homogeneous = points_camera @ intrinsic.T
    pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
    if not np.isfinite(pixels).all():
        raise ValueError("Projected pixel coordinates must be finite.")
    return pixels


def _write_json(path: Path, value: object) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")


def _load_json_object(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Expected an ordinary JSON file: {path}")
    try:
        raw: object = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON at {path}: {error}") from error
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise TypeError(f"JSON document must be an object with string keys: {path}")
    return cast(dict[str, object], raw)


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _require_staging_directory(path: Path) -> None:
    if not path.is_dir() or path.is_symlink():
        raise ValueError(
            f"Alignment staging must be an ordinary existing directory: {path}"
        )


def _require_output_directory(path: Path) -> None:
    if not path.is_dir() or path.is_symlink():
        raise ValueError(
            f"Alignment output must be an ordinary existing directory: {path}"
        )


__all__ = [
    "ALIGNMENT_FILE",
    "COURT_GEOMETRY_FILE",
    "DIAGNOSTICS_DIRECTORY",
    "GROUND_LINE_MAP_FILE",
    "LINE_HEATMAP_DIRECTORY",
    "load_accepted_layout",
    "load_alignment_evidence",
    "load_alignment_result",
    "validate_alignment_outputs",
    "validate_court_transform_binding",
    "validate_projection_equivalence",
    "write_alignment_outputs",
]
