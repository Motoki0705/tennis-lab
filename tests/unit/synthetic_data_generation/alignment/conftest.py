"""Focused semantic alignment fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from src.synthetic_data_generation.alignment.contracts import (
    AlignmentAcceptancePolicy,
    AlignmentEvaluationDiagnostics,
    AlignmentEvaluationOutcome,
    AlignmentEvaluationPolicy,
    AlignmentEvidence,
    AlignmentEvidenceDiagnostics,
    AlignmentPartitions,
    AlignmentTrace,
    AlignmentTraceCandidateState,
    AlignmentTracePhase,
    AlignmentTraceStep,
    CameraLineDiagnostics,
    CameraOwnershipRule,
    CameraSelectionPolicy,
    CandidateEvidence,
    CandidateScaleDiagnostics,
    CorrespondenceSet,
    FixedCameraSelectionDiagnostics,
    GroundPlaneFrame,
    LineInferenceDeterminismDiagnostics,
    MeasuredCameraLines,
    MetricSceneAdapter,
    PartitionThresholds,
    ProposalSearchDiagnostics,
)
from src.synthetic_data_generation.alignment.settings import WholeCourtEvidenceSettings
from src.synthetic_data_generation.scene_contract import RigidTransform
from src.utils.schema.court import HALF_DOUBLES_WIDTH, HALF_LENGTH


def _rigid(*, angle: float, translation: tuple[float, float, float]) -> RigidTransform:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    matrix = np.asarray(
        [
            [cosine, -sine, 0.0, translation[0]],
            [sine, cosine, 0.0, translation[1]],
            [0.0, 0.0, 1.0, translation[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return RigidTransform.from_matrix(matrix)


def _candidate(
    *,
    court_id: str,
    candidate_id: str,
    transform: RigidTransform,
) -> CandidateEvidence:
    longitudinal = np.asarray(
        [
            (offset, tangential, 0.0)
            for offset in (-HALF_DOUBLES_WIDTH, HALF_DOUBLES_WIDTH)
            for tangential in np.linspace(-10.0, 10.0, 41)
        ],
        dtype=np.float64,
    )
    transverse = np.asarray(
        [
            (tangential, offset, 0.0)
            for offset in (-HALF_LENGTH, HALF_LENGTH)
            for tangential in np.linspace(-4.63, 4.63, 31)
        ],
        dtype=np.float64,
    )
    court = np.concatenate((longitudinal, transverse))
    fit_court = np.concatenate((court, court))
    holdout_court = np.concatenate((court, court))
    fit_camera_ids = ("fit-0",) * len(court) + ("fit-1",) * len(court)
    holdout_camera_ids = ("holdout-0",) * len(court) + ("holdout-1",) * len(court)
    return CandidateEvidence(
        court_instance_id=court_id,
        candidate_id=candidate_id,
        fit=CorrespondenceSet(
            points_court=fit_court,
            points_scene=transform.apply(fit_court),
            camera_ids=fit_camera_ids,
        ),
        holdout=CorrespondenceSet(
            points_court=holdout_court,
            points_scene=transform.apply(holdout_court),
            camera_ids=holdout_camera_ids,
        ),
    )


@pytest.fixture
def alignment_policy() -> AlignmentAcceptancePolicy:
    """Return separate strict fit and holdout gates."""
    fit = PartitionThresholds(
        minimum_camera_count=2,
        minimum_correspondence_count=6,
        inlier_distance_m=0.01,
        minimum_inlier_fraction=1.0,
        maximum_rms_error_m=0.01,
        maximum_q95_error_m=0.01,
    )
    holdout = PartitionThresholds(
        minimum_camera_count=2,
        minimum_correspondence_count=6,
        inlier_distance_m=0.02,
        minimum_inlier_fraction=1.0,
        maximum_rms_error_m=0.02,
        maximum_q95_error_m=0.02,
    )
    return AlignmentAcceptancePolicy(fit=fit, holdout=holdout)


@pytest.fixture
def alignment_evidence() -> AlignmentEvidence:
    """Return two independently accepted physical courts."""
    first = _candidate(
        court_id="court-a",
        candidate_id="candidate-a",
        transform=_rigid(angle=0.2, translation=(1.0, 2.0, 0.0)),
    )
    second = _candidate(
        court_id="court-b",
        candidate_id="candidate-b",
        transform=_rigid(angle=-0.15, translation=(20.0, 1.0, 0.0)),
    )
    whole_court_settings = _whole_court_settings()
    return AlignmentEvidence(
        partitions=AlignmentPartitions(
            fit_camera_ids=("fit-0", "fit-1"),
            holdout_camera_ids=("holdout-0", "holdout-1"),
        ),
        candidates=(first, second),
        measured_camera_lines=tuple(
            MeasuredCameraLines(
                camera_id=camera_id,
                points_nht_scene=np.column_stack(
                    (
                        np.linspace(-1.0, 1.0, 80),
                        np.linspace(1.0, -1.0, 80),
                        np.zeros(80, dtype=np.float64),
                    )
                ),
            )
            for index, camera_id in enumerate(
                ("fit-0", "fit-1", "holdout-0", "holdout-1")
            )
        ),
        complex_points_scene=np.asarray(
            [[-8.0, -15.0, -1.0], [30.0, 16.0, 5.0]], dtype=np.float64
        ),
        primary_candidate_id="candidate-a",
        metric_adapter=MetricSceneAdapter.from_nht_scene_from_metric_scene(
            np.eye(4, dtype=np.float64)
        ),
        diagnostics=AlignmentEvidenceDiagnostics(
            cameras=tuple(
                CameraLineDiagnostics(
                    camera_id=camera_id,
                    selected_line_pixel_count=100,
                    projected_line_point_count=80,
                )
                for camera_id in ("fit-0", "fit-1", "holdout-0", "holdout-1")
            ),
            candidate_scales=(
                CandidateScaleDiagnostics(
                    candidate_id="candidate-a",
                    nht_scene_units_per_metre=1.0,
                    template_score=0.8,
                    common_scale_refit_template_score=0.82,
                    common_scale_refit_center_uv_metres=(1.0, 2.0),
                    common_scale_refit_orientation_radians=0.2,
                    common_scale_refit_center_displacement_metres=0.0,
                    maximum_common_scale_refit_center_displacement_metres=(
                        whole_court_settings.maximum_center_refit_displacement_metres
                    ),
                    proposal_orientation_band_minimum_radians=-0.5,
                    proposal_orientation_band_maximum_radians=0.5,
                    proposal_residual_point_count_before_suppression=100,
                    proposal_residual_point_count_after_suppression=50,
                    native_center_uv_metres=(1.0, 2.0),
                    native_orientation_radians=0.2,
                ),
                CandidateScaleDiagnostics(
                    candidate_id="candidate-b",
                    nht_scene_units_per_metre=1.0,
                    template_score=0.7,
                    common_scale_refit_template_score=0.72,
                    common_scale_refit_center_uv_metres=(20.0, 1.0),
                    common_scale_refit_orientation_radians=-0.15,
                    common_scale_refit_center_displacement_metres=0.0,
                    maximum_common_scale_refit_center_displacement_metres=(
                        whole_court_settings.maximum_center_refit_displacement_metres
                    ),
                    proposal_orientation_band_minimum_radians=-0.5,
                    proposal_orientation_band_maximum_radians=0.5,
                    proposal_residual_point_count_before_suppression=50,
                    proposal_residual_point_count_after_suppression=25,
                    native_center_uv_metres=(20.0, 1.0),
                    native_orientation_radians=-0.15,
                ),
            ),
            common_nht_scene_units_per_metre=1.0,
            maximum_relative_scale_deviation=0.0,
            selection=FixedCameraSelectionDiagnostics(
                policy=CameraSelectionPolicy.NESTED_UNIFORM_PREFIX_V1,
                ownership_rule=(
                    CameraOwnershipRule.FIXED_UNIT_EVEN_HOLDOUT_SLOTS_V1
                ),
                requested_camera_count=4,
                available_camera_count=4,
                candidate_count=2,
                orientation_family_count=1,
                fit_cameras_per_unit=1,
                holdout_cameras_per_unit=1,
                camera_prefix_ids=(
                    "fit-0",
                    "holdout-0",
                    "fit-1",
                    "holdout-1",
                ),
                fit_camera_ids=("fit-0", "fit-1"),
                holdout_camera_ids=("holdout-0", "holdout-1"),
                observed_camera_ids=(
                    "fit-0",
                    "holdout-0",
                    "fit-1",
                    "holdout-1",
                ),
                excluded_cameras=(),
            ),
            evaluation=AlignmentEvaluationDiagnostics(
                policy=(
                    AlignmentEvaluationPolicy.FIT_SELECT_ONCE_HOLDOUT_EVALUATE_ONCE_V1
                ),
                evaluation_index=0,
                fit_camera_ids=("fit-0", "fit-1"),
                holdout_camera_ids=("holdout-0", "holdout-1"),
                candidate_ids=("candidate-a", "candidate-b"),
                fit_evaluation_count=1,
                holdout_evaluation_count=1,
                outcome=AlignmentEvaluationOutcome.FULL_VALIDATION_PASS,
            ),
            determinism=LineInferenceDeterminismDiagnostics(
                seed=42,
                device="cpu",
                model_eval=True,
                inference_mode=True,
                deterministic_algorithms=True,
                deterministic_warn_only=False,
                cudnn_benchmark=False,
                cudnn_deterministic=True,
                cuda_matmul_allow_tf32=False,
                cudnn_allow_tf32=False,
                cublas_workspace_config=None,
                torch_version="test",
                cuda_version=None,
                device_name="cpu",
                cross_hardware_bit_identity_claimed=False,
            ),
            proposal_search=ProposalSearchDiagnostics(
                orientation_band_count=1,
                center_tile_count=1,
                maximum_center_tile_width_scene_units=1.0,
                maximum_complete_branch_count=1,
                maximum_tile_state_count=2,
                maximum_residual_state_count=2,
                residual_state_count=2,
                residual_tree_build_count=2,
                explored_tile_state_count=2,
                geometrically_impossible_tile_state_count=0,
                feasible_proposal_count_before_deduplication=2,
                duplicate_proposal_count=0,
                retained_proposal_count=2,
                expanded_state_count=2,
                feasible_complete_state_count=1,
                selected_orientation_band_indices=(0, 0),
                selected_center_tile_indices=(0, 0),
                original_point_count=100,
                selected_residual_point_count=25,
                selected_explained_point_count=75,
                selected_native_score_sum=1.5,
            ),
            excluded_cameras=(),
            ground_plane_frame=GroundPlaneFrame(
                origin_metric_scene=(0.0, 0.0, 0.0),
                basis_u_metric_scene=(1.0, 0.0, 0.0),
                basis_v_metric_scene=(0.0, 1.0, 0.0),
                normal_metric_scene=(0.0, 0.0, 1.0),
                bounds_uv_metres=(-10.0, 35.0, -20.0, 20.0),
            ),
            alignment_trace=_alignment_trace(),
        ),
        whole_court_settings=whole_court_settings,
    )


def _alignment_trace() -> AlignmentTrace:
    candidate_ids = ("candidate-a", "candidate-b")
    phase_values = (
        (
            AlignmentTracePhase.PROPOSAL_SELECTION,
            ((0.8, 2.2, 0.18, 0.75), (19.8, 1.2, -0.12, 0.65)),
        ),
        (
            AlignmentTracePhase.NATIVE_REFINEMENT,
            ((1.0, 2.0, 0.2, 0.8), (20.0, 1.0, -0.15, 0.7)),
        ),
        (
            AlignmentTracePhase.COMMON_SCALE_REFIT,
            ((1.0, 2.0, 0.2, 0.82), (20.0, 1.0, -0.15, 0.72)),
        ),
        (
            AlignmentTracePhase.FINAL_ALIGNMENT,
            ((1.0, 2.0, 0.2, 0.82), (20.0, 1.0, -0.15, 0.72)),
        ),
    )
    residual_counts = ((100, 50), (50, 25))
    steps = tuple(
        AlignmentTraceStep(
            step_index=step_index,
            phase=phase,
            candidates=tuple(
                AlignmentTraceCandidateState(
                    candidate_id=candidate_id,
                    center_uv_metres=(values[0], values[1]),
                    orientation_radians=values[2],
                    nht_scene_units_per_metre=1.0,
                    template_score=values[3],
                    orientation_band_index=0,
                    center_tile_index=0,
                    residual_point_count_before_suppression=residual_counts[index][0],
                    residual_point_count_after_suppression=residual_counts[index][1],
                )
                for index, (candidate_id, values) in enumerate(
                    zip(candidate_ids, states, strict=True)
                )
            ),
            score_sum=sum(values[3] for values in states),
        )
        for step_index, (phase, states) in enumerate(phase_values)
    )
    return AlignmentTrace(final_candidate_ids=candidate_ids, steps=steps)


def _whole_court_settings() -> WholeCourtEvidenceSettings:
    maximum_scale_deviation = 0.07290400972053462
    localization_tolerance = 0.3
    maximum_center_displacement = (
        maximum_scale_deviation * np.hypot(HALF_DOUBLES_WIDTH, HALF_LENGTH)
        + localization_tolerance
    )
    return WholeCourtEvidenceSettings(
        required_court_count=2,
        maximum_common_scale_relative_deviation=maximum_scale_deviation,
        maximum_center_refit_displacement_metres=maximum_center_displacement,
        minimum_distinct_offset_levels=2,
        minimum_matches_per_offset_level=3,
        minimum_level_camera_count=2,
        minimum_secondary_tangential_span_metres=0.6,
        minimum_longitudinal_offset_span_metres=8.23,
        minimum_longitudinal_tangential_span_metres=12.8,
        minimum_transverse_offset_span_metres=12.8,
        minimum_transverse_tangential_span_metres=8.23,
        samples_per_metre=3.0,
        inlier_distance_metres=localization_tolerance,
        minimum_inlier_fraction=0.9,
        maximum_q95_error_metres=0.1,
        minimum_semantic_segment_inlier_fraction=0.8,
        minimum_center_separation_metres=10.97,
        maximum_footprint_overlap_fraction=1.0e-9,
    )
