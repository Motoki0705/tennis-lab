"""Tests for strict fit/holdout and multi-court alignment contracts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from src.synthetic_data_generation.alignment.contracts import (
    AlignmentAcceptancePolicy,
    AlignmentEvidence,
    AlignmentPartitions,
    AlignmentResult,
    AlignmentTrace,
    AlignmentTracePhase,
    CameraOwnershipRule,
    CameraSelectionPolicy,
    CandidateEvidence,
    CandidateScaleDiagnostics,
    CorrespondenceSet,
    FixedCameraSelectionDiagnostics,
    GroundPlaneFrame,
    MeasuredCameraLines,
    MetricSceneAdapter,
)
from src.synthetic_data_generation.alignment.fitting import (
    fit_alignment,
    fit_rigid_transform,
    whole_court_diagnostics,
)
from src.synthetic_data_generation.alignment.settings import WholeCourtEvidenceSettings
from src.synthetic_data_generation.alignment.whole_court import (
    evaluate_court_identifiability,
    evaluate_court_topology,
    sample_court_line_segments,
)
from src.synthetic_data_generation.scene_contract import RigidTransform
from src.utils.schema.court import (
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    HALF_SINGLES_WIDTH,
    SERVICE_LINE_DISTANCE,
)


def test_fixed_48_camera_ownership_rejects_contiguous_fit_holdout_claim() -> None:
    prefix = tuple(f"c{index:02d}" for index in range(48))
    holdout_slots = {1, 4, 7, 10}
    expected_fit = tuple(
        camera_id
        for index, camera_id in enumerate(prefix)
        if index % 12 not in holdout_slots
    )
    expected_holdout = tuple(
        camera_id
        for index, camera_id in enumerate(prefix)
        if index % 12 in holdout_slots
    )
    selection = FixedCameraSelectionDiagnostics(
        policy=CameraSelectionPolicy.NESTED_UNIFORM_PREFIX_V1,
        ownership_rule=CameraOwnershipRule.FIXED_UNIT_EVEN_HOLDOUT_SLOTS_V1,
        requested_camera_count=48,
        available_camera_count=491,
        candidate_count=2,
        orientation_family_count=2,
        fit_cameras_per_unit=8,
        holdout_cameras_per_unit=4,
        camera_prefix_ids=prefix,
        fit_camera_ids=expected_fit,
        holdout_camera_ids=expected_holdout,
        observed_camera_ids=prefix,
        excluded_cameras=(),
    )

    assert FixedCameraSelectionDiagnostics.from_dict(selection.to_dict()) == selection
    with pytest.raises(ValueError, match="unit slot rule"):
        replace(
            selection,
            fit_camera_ids=prefix[:32],
            holdout_camera_ids=prefix[32:],
        )


def test_all_accepted_courts_have_unique_ids_bounds_and_reciprocal_transforms(
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    result = fit_alignment(alignment_evidence, policy=alignment_policy)

    assert [court.court_instance_id for court in result.layout.courts] == [
        "court-a",
        "court-b",
    ]
    assert [court.candidate_id for court in result.layout.courts] == [
        "candidate-a",
        "candidate-b",
    ]
    assert result.layout.primary_court_instance_id == "court-a"
    assert result.layout.complex_bounds_scene == (-8.0, -15.0, -1.0, 30.0, 16.0, 5.0)
    for court in result.layout.courts:
        product = court.court_from_scene.matrix() @ court.scene_from_court.matrix()
        reverse = court.scene_from_court.matrix() @ court.court_from_scene.matrix()
        np.testing.assert_allclose(product, np.eye(4), atol=1.0e-7)
        np.testing.assert_allclose(reverse, np.eye(4), atol=1.0e-7)
        assert court.fit_status == "accepted"
        assert court.holdout_status == "accepted"
        assert court.fit_metrics["camera_ids"] == ["fit-0", "fit-1"]
        assert court.holdout_metrics["camera_ids"] == ["holdout-0", "holdout-1"]


def test_complex_bounds_reject_isolated_sparse_cloud_outliers(
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    support = np.column_stack(
        (
            np.linspace(-20.0, 30.0, 200),
            np.linspace(-15.0, 16.0, 200),
            np.linspace(-1.0, 8.0, 200),
        )
    )
    support = np.vstack(
        (
            support,
            np.asarray(((-10_000.0, -20_000.0, -5_000.0),)),
            np.asarray(((30_000.0, 40_000.0, 6_000.0),)),
        )
    )

    result = fit_alignment(
        replace(alignment_evidence, complex_points_scene=support),
        policy=alignment_policy,
    )

    bounds = np.asarray(result.layout.complex_bounds_scene).reshape(2, 3)
    assert np.all(np.abs(bounds) < 100.0)


def test_holdout_is_evaluated_without_refitting_and_rejection_fails_closed(
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    first, second = alignment_evidence.candidates
    bad_holdout = CorrespondenceSet(
        points_court=first.holdout.points_court,
        points_scene=first.holdout.points_scene + np.asarray([0.5, 0.0, 0.0]),
        camera_ids=first.holdout.camera_ids,
    )
    rejected = CandidateEvidence(
        court_instance_id=first.court_instance_id,
        candidate_id=first.candidate_id,
        fit=first.fit,
        holdout=bad_holdout,
    )
    evidence = AlignmentEvidence(
        partitions=alignment_evidence.partitions,
        candidates=(rejected, second),
        measured_camera_lines=alignment_evidence.measured_camera_lines,
        complex_points_scene=alignment_evidence.complex_points_scene,
        primary_candidate_id="candidate-b",
        metric_adapter=alignment_evidence.metric_adapter,
        diagnostics=alignment_evidence.diagnostics,
        whole_court_settings=alignment_evidence.whole_court_settings,
    )

    baseline_transform = fit_rigid_transform(first.fit)
    rejected_transform = fit_rigid_transform(rejected.fit)
    np.testing.assert_allclose(
        rejected_transform.matrix(), baseline_transform.matrix(), atol=1.0e-10
    )
    with pytest.raises(ValueError, match='selected_correspondence_accepted":false'):
        fit_alignment(evidence, policy=alignment_policy)


def test_all_holdout_rejections_fail_closed(
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
            points_scene=first.holdout.points_scene + 1.0,
            camera_ids=first.holdout.camera_ids,
        ),
    )
    second = alignment_evidence.candidates[1]
    second_rejected = replace(
        second,
        holdout=CorrespondenceSet(
            points_court=second.holdout.points_court,
            points_scene=second.holdout.points_scene + 1.0,
            camera_ids=second.holdout.camera_ids,
        ),
    )
    evidence = AlignmentEvidence(
        partitions=alignment_evidence.partitions,
        candidates=(rejected, second_rejected),
        measured_camera_lines=alignment_evidence.measured_camera_lines,
        complex_points_scene=alignment_evidence.complex_points_scene,
        primary_candidate_id=None,
        metric_adapter=alignment_evidence.metric_adapter,
        diagnostics=alignment_evidence.diagnostics,
        whole_court_settings=alignment_evidence.whole_court_settings,
    )

    with pytest.raises(ValueError, match='selected_correspondence_accepted":false'):
        fit_alignment(evidence, policy=alignment_policy)


def test_alignment_serialization_is_strict_and_recomputes_status(
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    result = fit_alignment(alignment_evidence, policy=alignment_policy)
    payload = result.to_dict()

    loaded = AlignmentResult.from_dict(payload)
    assert loaded.to_dict() == payload

    with_unknown = deepcopy(payload)
    with_unknown["artifact_fingerprint"] = "forbidden"
    with pytest.raises(ValueError, match="unknown"):
        AlignmentResult.from_dict(with_unknown)

    bad_status = deepcopy(payload)
    candidates = bad_status["candidates"]
    assert isinstance(candidates, list)
    candidates[0]["holdout"]["status"] = "rejected"
    with pytest.raises(ValueError, match="status disagrees"):
        AlignmentResult.from_dict(bad_status)


def test_partitions_reject_overlap_and_candidate_partition_drift(
    alignment_evidence: AlignmentEvidence,
) -> None:
    with pytest.raises(ValueError, match="overlap"):
        AlignmentPartitions(
            fit_camera_ids=("camera-a",),
            holdout_camera_ids=("camera-a",),
        )

    first = alignment_evidence.candidates[0]
    drifted = CandidateEvidence(
        court_instance_id=first.court_instance_id,
        candidate_id=first.candidate_id,
        fit=CorrespondenceSet(
            points_court=first.fit.points_court,
            points_scene=first.fit.points_scene,
            camera_ids=("outside-fit",) * len(first.fit.points_court),
        ),
        holdout=first.holdout,
    )
    with pytest.raises(ValueError, match="outside its declared partition"):
        AlignmentEvidence(
            partitions=alignment_evidence.partitions,
            candidates=(drifted, alignment_evidence.candidates[1]),
            measured_camera_lines=alignment_evidence.measured_camera_lines,
            complex_points_scene=alignment_evidence.complex_points_scene,
            primary_candidate_id=None,
            metric_adapter=alignment_evidence.metric_adapter,
            diagnostics=alignment_evidence.diagnostics,
            whole_court_settings=alignment_evidence.whole_court_settings,
        )


def test_metric_scene_adapter_keeps_nht_similarity_outside_rigid_court_contracts() -> (
    None
):
    nht_from_metric = np.asarray(
        [
            [0.0, -0.07, 0.0, 1.0],
            [0.07, 0.0, 0.0, -2.0],
            [0.0, 0.0, 0.07, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    adapter = MetricSceneAdapter.from_nht_scene_from_metric_scene(nht_from_metric)
    points_metric = np.asarray([[1.0, 2.0, 3.0], [-4.0, 5.0, 0.0]])
    points_nht = adapter.nht_from_metric_points(points_metric)

    np.testing.assert_allclose(
        adapter.metric_from_nht_points(points_nht), points_metric
    )
    assert adapter.nht_scene_units_per_metre == pytest.approx(0.07)

    camera_matrix = np.eye(4, dtype=np.float64)
    camera_matrix[:3, 3] = (4.0, -3.0, 2.0)
    camera_metric = RigidTransform.from_matrix(camera_matrix)
    camera_nht = adapter.nht_from_metric_camera(camera_metric)
    round_trip = adapter.metric_from_nht_camera(camera_nht)
    np.testing.assert_allclose(round_trip.matrix(), camera_metric.matrix(), atol=1.0e-9)
    np.testing.assert_allclose(
        round_trip.matrix()[:3, :3].T @ round_trip.matrix()[:3, :3],
        np.eye(3),
        atol=1.0e-9,
    )

    non_uniform = nht_from_metric.copy()
    non_uniform[2, 2] = 0.08
    with pytest.raises(ValueError, match="uniform scale"):
        MetricSceneAdapter.from_nht_scene_from_metric_scene(non_uniform)


def test_metric_ground_plane_frame_round_trip_and_adapter_binding() -> None:
    nht_from_metric = np.asarray(
        [
            [0.0, -0.07, 0.0, 1.0],
            [0.07, 0.0, 0.0, -2.0],
            [0.0, 0.0, 0.07, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    adapter = MetricSceneAdapter.from_nht_scene_from_metric_scene(nht_from_metric)
    frame = GroundPlaneFrame.from_nht_frame(
        metric_adapter=adapter,
        origin_nht_scene=np.asarray((1.0, -2.0, 0.5), dtype=np.float64),
        basis_u_nht_scene=np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
        basis_v_nht_scene=np.asarray((0.0, 1.0, 0.0), dtype=np.float64),
        normal_nht_scene=np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
        bounds_uv_nht_scene=np.asarray(
            (-0.7, 1.4, -0.35, 0.7),
            dtype=np.float64,
        ),
    )
    uv = np.asarray(((-2.0, 3.0), (5.0, -1.0)), dtype=np.float64)

    np.testing.assert_allclose(frame.to_uv(frame.from_uv(uv)), uv, atol=1.0e-10)
    assert frame.bounds_uv_metres == pytest.approx((-10.0, 20.0, -5.0, 10.0))
    assert GroundPlaneFrame.from_dict(frame.to_dict()) == frame
    basis = np.column_stack(
        (
            frame.basis_u_metric_scene,
            frame.basis_v_metric_scene,
            frame.normal_metric_scene,
        )
    )
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1.0e-9)
    assert np.linalg.det(basis) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    (
        (
            "basis_u_nht_scene",
            np.asarray(((1.0, 0.0, 0.0),), dtype=np.float64),
            ValueError,
            "one-dimensional",
        ),
        (
            "bounds_uv_nht_scene",
            np.asarray((-1.0, 1.0, -1.0), dtype=np.float64),
            ValueError,
            "exactly 4",
        ),
        (
            "basis_v_nht_scene",
            np.asarray(("x", "y", "z")),
            TypeError,
            "real numeric dtype",
        ),
        (
            "normal_nht_scene",
            np.asarray((0.0, 1.0, object()), dtype=object),
            TypeError,
            "real numeric dtype",
        ),
        (
            "origin_nht_scene",
            np.asarray((0.0, np.nan, 0.0), dtype=np.float64),
            ValueError,
            "finite",
        ),
        (
            "bounds_uv_nht_scene",
            np.asarray((-1.0, 1.0, -np.inf, 1.0), dtype=np.float64),
            ValueError,
            "finite",
        ),
        (
            "basis_u_nht_scene",
            (1.0, [0.0], 0.0),
            TypeError,
            "numeric",
        ),
    ),
)
def test_metric_ground_plane_frame_rejects_invalid_nht_vectors(
    field: str,
    value: object,
    error: type[Exception],
    message: str,
) -> None:
    adapter = MetricSceneAdapter.from_nht_scene_from_metric_scene(
        np.eye(4, dtype=np.float64)
    )
    values: dict[str, object] = {
        "origin_nht_scene": np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
        "basis_u_nht_scene": np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
        "basis_v_nht_scene": np.asarray((0.0, 1.0, 0.0), dtype=np.float64),
        "normal_nht_scene": np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
        "bounds_uv_nht_scene": np.asarray(
            (-1.0, 1.0, -1.0, 1.0),
            dtype=np.float64,
        ),
    }
    values[field] = value

    with pytest.raises(error, match=message):
        GroundPlaneFrame.from_nht_frame(
            metric_adapter=adapter,
            origin_nht_scene=values["origin_nht_scene"],  # type: ignore[arg-type]
            basis_u_nht_scene=values["basis_u_nht_scene"],  # type: ignore[arg-type]
            basis_v_nht_scene=values["basis_v_nht_scene"],  # type: ignore[arg-type]
            normal_nht_scene=values["normal_nht_scene"],  # type: ignore[arg-type]
            bounds_uv_nht_scene=values["bounds_uv_nht_scene"],  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("basis_v_metric_scene", (1.0, 0.0, 0.0), "orthonormal"),
        ("normal_metric_scene", (0.0, 0.0, -1.0), "right-handed"),
        ("origin_metric_scene", (0.0, np.nan, 0.0), "finite"),
        ("bounds_uv_metres", (0.0, 0.0, -1.0, 1.0), "positive area"),
    ),
)
def test_metric_ground_plane_frame_rejects_invalid_geometry(
    field: str,
    value: tuple[float, ...],
    message: str,
) -> None:
    payload: dict[str, object] = {
        "origin_metric_scene": (0.0, 0.0, 0.0),
        "basis_u_metric_scene": (1.0, 0.0, 0.0),
        "basis_v_metric_scene": (0.0, 1.0, 0.0),
        "normal_metric_scene": (0.0, 0.0, 1.0),
        "bounds_uv_metres": (-1.0, 1.0, -1.0, 1.0),
    }
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        GroundPlaneFrame(**payload)  # type: ignore[arg-type]


def test_alignment_trace_round_trip_rejects_reordering_and_score_mismatch(
    alignment_evidence: AlignmentEvidence,
) -> None:
    payload = alignment_evidence.alignment_trace.to_dict()

    assert AlignmentTrace.from_dict(payload) == alignment_evidence.alignment_trace

    reordered = deepcopy(payload)
    steps = cast(list[dict[str, object]], reordered["steps"])
    steps[1], steps[2] = steps[2], steps[1]
    with pytest.raises(ValueError, match="reordered"):
        AlignmentTrace.from_dict(reordered)

    mismatched_score = deepcopy(payload)
    score_steps = cast(list[dict[str, object]], mismatched_score["steps"])
    score_steps[0]["score_sum"] = 123.0
    with pytest.raises(ValueError, match="score_sum"):
        AlignmentTrace.from_dict(mismatched_score)

    non_finite = deepcopy(payload)
    finite_steps = cast(list[dict[str, Any]], non_finite["steps"])
    finite_steps[0]["candidates"][0]["template_score"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        AlignmentTrace.from_dict(non_finite)


def test_final_trace_is_bound_to_recomputed_alignment(
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    final_step = alignment_evidence.alignment_trace.step(
        AlignmentTracePhase.FINAL_ALIGNMENT
    )
    drifted_state = replace(
        final_step.candidates[0],
        center_uv_metres=(1.2, 2.0),
    )
    drifted_trace = replace(
        alignment_evidence.alignment_trace,
        steps=(
            *alignment_evidence.alignment_trace.steps[:-1],
            replace(
                final_step,
                candidates=(drifted_state, *final_step.candidates[1:]),
            ),
        ),
    )
    evidence = replace(
        alignment_evidence,
        diagnostics=replace(
            alignment_evidence.diagnostics,
            alignment_trace=drifted_trace,
        ),
    )

    with pytest.raises(ValueError, match="center disagrees with the trace"):
        fit_alignment(evidence, policy=alignment_policy)


def test_common_scale_refit_diagnostics_reject_center_bound_violation() -> None:
    with pytest.raises(ValueError, match="exceeds its derived maximum"):
        CandidateScaleDiagnostics(
            candidate_id="candidate-a",
            nht_scene_units_per_metre=0.07,
            template_score=0.8,
            common_scale_refit_template_score=0.81,
            common_scale_refit_center_uv_metres=(0.0, 0.0),
            common_scale_refit_orientation_radians=0.0,
            common_scale_refit_center_displacement_metres=1.01,
            maximum_common_scale_refit_center_displacement_metres=1.0,
            proposal_orientation_band_minimum_radians=-0.5,
            proposal_orientation_band_maximum_radians=0.5,
            proposal_residual_point_count_before_suppression=100,
            proposal_residual_point_count_after_suppression=50,
            native_center_uv_metres=(0.0, 0.0),
            native_orientation_radians=0.0,
        )


def test_partial_two_family_geometry_passes_despite_missing_template_segments(
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    evidence = _with_whole_court_evidence(
        alignment_evidence,
        diagnostic_segment_names={
            "doubles_sideline_left",
            "doubles_sideline_right",
            "baseline_near",
            "baseline_far",
        },
    )

    result = fit_alignment(evidence, policy=alignment_policy)
    diagnostics = whole_court_diagnostics(
        evidence,
        candidates=result.candidates,
        policy=alignment_policy,
    )

    assert len(result.layout.courts) == 2
    assert diagnostics is not None
    candidate_diagnostics = cast(list[dict[str, Any]], diagnostics["candidates"])
    first = candidate_diagnostics[0]
    assert first["fit"]["identifiability"]["accepted"] is True
    assert not all(
        first["fit"]["whole_template_diagnostic"][
            "diagnostic_threshold_checks"
        ].values()
    )


def test_only_parallel_lines_fail_even_when_whole_template_diagnostics_pass(
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    parallel = _longitudinal_points(
        offsets=(-HALF_DOUBLES_WIDTH, HALF_DOUBLES_WIDTH),
        tangential_limit=10.0,
    )
    evidence = _with_whole_court_evidence(
        alignment_evidence,
        fit_points_court=parallel,
        holdout_points_court=parallel,
    )

    with pytest.raises(ValueError, match='"transverse"'):
        fit_alignment(evidence, policy=alignment_policy)


def test_insufficient_semantic_offset_span_fails(
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    narrow_longitudinal = _identifiable_points(
        longitudinal_offsets=(-HALF_DOUBLES_WIDTH, -HALF_SINGLES_WIDTH),
    )
    evidence = _with_whole_court_evidence(
        alignment_evidence,
        fit_points_court=narrow_longitudinal,
        holdout_points_court=narrow_longitudinal,
    )

    with pytest.raises(ValueError, match="minimum_offset_separation_metres"):
        fit_alignment(evidence, policy=alignment_policy)


def test_insufficient_tangential_span_fails(
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    short_longitudinal = _identifiable_points(longitudinal_tangential_limit=5.0)
    evidence = _with_whole_court_evidence(
        alignment_evidence,
        fit_points_court=short_longitudinal,
        holdout_points_court=short_longitudinal,
    )

    with pytest.raises(ValueError, match="anchor_level_eligible"):
        fit_alignment(evidence, policy=alignment_policy)


def test_fit_only_identifiability_cannot_substitute_for_holdout(
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    parallel = _longitudinal_points(
        offsets=(-HALF_DOUBLES_WIDTH, HALF_DOUBLES_WIDTH),
        tangential_limit=10.0,
    )
    evidence = _with_whole_court_evidence(
        alignment_evidence,
        fit_points_court=_identifiable_points(),
        holdout_points_court=parallel,
    )

    with pytest.raises(ValueError, match='"holdout"'):
        fit_alignment(evidence, policy=alignment_policy)


def test_long_anchor_and_independently_supported_secondary_pass() -> None:
    correspondences = _camera_repeated_correspondences(
        _anchor_secondary_points(),
        camera_count=6,
    )

    metrics = _identifiability_payload(
        correspondences,
        minimum_camera_count=6,
    )

    assert metrics["accepted"] is True
    expected_pairs = {
        "longitudinal": (-HALF_DOUBLES_WIDTH, HALF_SINGLES_WIDTH),
        "transverse": (-HALF_LENGTH, SERVICE_LINE_DISTANCE),
    }
    for family_name, expected in expected_pairs.items():
        family = metrics[family_name]
        pair = family["qualifying_anchor_secondary_pair"]
        assert pair["accepted"] is True
        assert pair["rejection_reasons"] == []
        assert pair["anchor_offset_metres"] == pytest.approx(expected[0])
        assert pair["secondary_offset_metres"] == pytest.approx(expected[1])


def test_anchor_observed_by_one_camera_fails_multiview_support() -> None:
    anchor, secondary = _anchor_secondary_components()
    parts = [np.concatenate((anchor, secondary))]
    parts.extend(secondary for _index in range(5))
    correspondences = _partitioned_correspondences(parts)

    metrics = _identifiability_payload(
        correspondences,
        minimum_camera_count=6,
    )

    assert metrics["accepted"] is False
    for family_name in ("longitudinal", "transverse"):
        family = metrics[family_name]
        pair = family["qualifying_anchor_secondary_pair"]
        assert pair["accepted"] is False
        assert "anchor_level_eligible" in pair["rejection_reasons"]


def test_secondary_shorter_than_twice_localization_error_fails() -> None:
    correspondences = _camera_repeated_correspondences(
        _anchor_secondary_points(secondary_positions=(-0.2, 0.0, 0.2)),
        camera_count=6,
    )

    metrics = _identifiability_payload(
        correspondences,
        minimum_camera_count=6,
    )

    assert metrics["accepted"] is False
    for family_name in ("longitudinal", "transverse"):
        pair = metrics[family_name]["qualifying_anchor_secondary_pair"]
        assert pair["accepted"] is False
        assert "secondary_level_eligible" in pair["rejection_reasons"]


@pytest.mark.parametrize(
    ("minimum_camera_count", "observed_camera_count"),
    ((6, 5), (3, 2)),
)
def test_family_camera_union_below_partition_gate_fails(
    minimum_camera_count: int,
    observed_camera_count: int,
) -> None:
    correspondences = _camera_repeated_correspondences(
        _anchor_secondary_points(),
        camera_count=observed_camera_count,
    )

    metrics = _identifiability_payload(
        correspondences,
        minimum_camera_count=minimum_camera_count,
    )

    assert metrics["accepted"] is False
    for family_name in ("longitudinal", "transverse"):
        pair = metrics[family_name]["qualifying_anchor_secondary_pair"]
        assert pair["threshold_checks"]["minimum_family_camera_count"] is False


def test_invalid_fragment_padding_cameras_do_not_satisfy_qualifying_pair_gate() -> None:
    qualifying = _anchor_secondary_points()
    invalid_padding = np.asarray(
        [(HALF_DOUBLES_WIDTH, value, 0.0) for value in (-0.1, 0.0, 0.1)]
        + [(value, HALF_LENGTH, 0.0) for value in (-0.1, 0.0, 0.1)],
        dtype=np.float64,
    )
    correspondences = _partitioned_correspondences(
        [qualifying, qualifying] + [invalid_padding for _index in range(4)]
    )

    metrics = _identifiability_payload(
        correspondences,
        minimum_camera_count=6,
    )

    assert metrics["accepted"] is False
    for family_name in ("longitudinal", "transverse"):
        family = metrics[family_name]
        pair = family["qualifying_anchor_secondary_pair"]
        assert len(family["camera_ids"]) == 6
        assert pair["camera_ids"] == ["camera-0", "camera-1"]
        assert family["qualifying_camera_ids"] == ["camera-0", "camera-1"]
        assert pair["threshold_checks"]["minimum_family_camera_count"] is False


def test_four_repeated_line_fragments_do_not_establish_identifiability() -> None:
    fragments = np.asarray(
        [
            (offset, tangential, 0.0)
            for offset in (-HALF_DOUBLES_WIDTH, HALF_DOUBLES_WIDTH)
            for tangential in (-0.1, 0.0, 0.1)
        ]
        + [
            (tangential, offset, 0.0)
            for offset in (-HALF_LENGTH, HALF_LENGTH)
            for tangential in (-0.1, 0.0, 0.1)
        ],
        dtype=np.float64,
    )
    repeated_per_camera = np.repeat(fragments, 40, axis=0)
    correspondences = CorrespondenceSet(
        points_court=np.concatenate((repeated_per_camera, repeated_per_camera)),
        points_scene=np.concatenate((repeated_per_camera, repeated_per_camera)),
        camera_ids=("camera-0",) * len(repeated_per_camera)
        + ("camera-1",) * len(repeated_per_camera),
    )

    metrics = _identifiability_payload(correspondences)

    assert metrics["accepted"] is False
    for family_name in ("longitudinal", "transverse"):
        family = metrics[family_name]
        assert family["supported_offset_level_count"] == 0
        observed = [level for level in family["offset_levels"] if level["match_count"]]
        assert len(observed) == 2
        assert all(level["match_count"] == 240 for level in observed)
        assert all(level["unique_template_sample_count"] == 3 for level in observed)
        assert all(
            level["threshold_checks"]["minimum_anchor_tangential_span_metres"] is False
            for level in observed
        )


def test_repeated_template_index_does_not_inflate_unique_support() -> None:
    unique_points = np.asarray(
        [
            (-HALF_DOUBLES_WIDTH, 0.0, 0.0),
            (HALF_DOUBLES_WIDTH, 0.0, 0.0),
            (0.0, -HALF_LENGTH, 0.0),
            (0.0, HALF_LENGTH, 0.0),
        ],
        dtype=np.float64,
    )
    repeated_per_camera = np.repeat(unique_points, 100, axis=0)
    correspondences = CorrespondenceSet(
        points_court=np.concatenate((repeated_per_camera, repeated_per_camera)),
        points_scene=np.concatenate((repeated_per_camera, repeated_per_camera)),
        camera_ids=("camera-0",) * len(repeated_per_camera)
        + ("camera-1",) * len(repeated_per_camera),
    )

    metrics = _identifiability_payload(correspondences)

    assert metrics["accepted"] is False
    for family_name in ("longitudinal", "transverse"):
        observed = [
            level
            for level in metrics[family_name]["offset_levels"]
            if level["match_count"]
        ]
        assert all(level["match_count"] == 200 for level in observed)
        assert all(level["unique_template_sample_count"] == 1 for level in observed)
        assert all(
            level["threshold_checks"]["minimum_secondary_unique_template_samples"]
            is False
            for level in observed
        )


def test_endpoint_only_support_fails_unique_bin_coverage() -> None:
    endpoints = np.asarray(
        [
            (offset, tangential, 0.0)
            for offset in (-HALF_DOUBLES_WIDTH, HALF_DOUBLES_WIDTH)
            for tangential in (-6.5, 6.5)
        ]
        + [
            (tangential, offset, 0.0)
            for offset in (-HALF_LENGTH, HALF_LENGTH)
            for tangential in (-4.2, 4.2)
        ],
        dtype=np.float64,
    )
    correspondences = _repeated_camera_correspondences(endpoints)

    metrics = _identifiability_payload(correspondences)

    assert metrics["accepted"] is False
    expected_bins = {"longitudinal": 22, "transverse": 14}
    for family_name, required_bin_count in expected_bins.items():
        observed = [
            level
            for level in metrics[family_name]["offset_levels"]
            if level["match_count"]
        ]
        assert all(
            level["threshold_checks"]["minimum_anchor_tangential_span_metres"] is True
            for level in observed
        )
        assert all(
            level["tangential_bin_size_metres"] == pytest.approx(0.6)
            for level in observed
        )
        assert all(
            level["required_unique_tangential_bin_count"] == required_bin_count
            for level in observed
        )
        assert all(level["unique_tangential_bin_count"] == 2 for level in observed)
        assert all(
            level["threshold_checks"]["minimum_anchor_unique_tangential_bins"] is False
            for level in observed
        )


def test_pooled_across_offset_levels_cannot_supply_per_level_span() -> None:
    first_fragment = np.column_stack(
        (
            np.full(5, -HALF_DOUBLES_WIDTH),
            np.linspace(-10.0, -9.0, 5),
            np.zeros(5),
        )
    )
    second_fragment = np.column_stack(
        (
            np.full(5, HALF_DOUBLES_WIDTH),
            np.linspace(9.0, 10.0, 5),
            np.zeros(5),
        )
    )
    points = np.concatenate((first_fragment, second_fragment, _transverse_points()))
    correspondences = _repeated_camera_correspondences(points)

    metrics = _identifiability_payload(correspondences)

    assert metrics["accepted"] is False
    assert metrics["transverse"]["accepted"] is True
    longitudinal = metrics["longitudinal"]
    observed = [
        level for level in longitudinal["offset_levels"] if level["match_count"]
    ]
    assert (
        max(point[1] for point in np.concatenate((first_fragment, second_fragment)))
        - min(point[1] for point in np.concatenate((first_fragment, second_fragment)))
        >= 12.8
    )
    assert all(
        level["tangential_span_metres"] == pytest.approx(1.0) for level in observed
    )
    assert all(level["anchor_eligible"] is False for level in observed)
    assert longitudinal["qualifying_anchor_secondary_pair"]["accepted"] is False


def test_camera_union_cannot_supply_per_level_camera_count() -> None:
    camera_zero = np.concatenate(
        (
            _longitudinal_points(
                offsets=(-HALF_DOUBLES_WIDTH,),
                tangential_limit=10.0,
            ),
            _transverse_level_points(offset=-HALF_LENGTH),
        )
    )
    camera_one = np.concatenate(
        (
            _longitudinal_points(
                offsets=(HALF_DOUBLES_WIDTH,),
                tangential_limit=10.0,
            ),
            _transverse_level_points(offset=HALF_LENGTH),
        )
    )
    points = np.concatenate((camera_zero, camera_one))
    correspondences = CorrespondenceSet(
        points_court=points,
        points_scene=points,
        camera_ids=("camera-0",) * len(camera_zero) + ("camera-1",) * len(camera_one),
    )

    metrics = _identifiability_payload(correspondences)

    assert metrics["accepted"] is False
    for family_name in ("longitudinal", "transverse"):
        family = metrics[family_name]
        observed = [level for level in family["offset_levels"] if level["match_count"]]
        assert (
            len({camera for level in observed for camera in level["camera_ids"]}) == 2
        )
        assert all(len(level["camera_ids"]) == 1 for level in observed)
        assert all(
            level["threshold_checks"]["minimum_multiview_camera_count"] is False
            for level in observed
        )
        assert family["supported_offset_level_count"] == 0


def test_persisted_common_scale_inconsistency_fails_identifiability_gate(
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    evidence = _with_whole_court_evidence(alignment_evidence)
    native_step = evidence.alignment_trace.step(
        AlignmentTracePhase.NATIVE_REFINEMENT
    )
    drifted_native_step = replace(
        native_step,
        candidates=(
            replace(native_step.candidates[0], nht_scene_units_per_metre=0.8),
            replace(native_step.candidates[1], nht_scene_units_per_metre=1.2),
        ),
    )
    drifted_trace = replace(
        evidence.alignment_trace,
        steps=(
            evidence.alignment_trace.steps[0],
            drifted_native_step,
            *evidence.alignment_trace.steps[2:],
        ),
    )
    diagnostics = replace(
        evidence.diagnostics,
        candidate_scales=(
            replace(
                evidence.diagnostics.candidate_scales[0],
                nht_scene_units_per_metre=0.8,
            ),
            replace(
                evidence.diagnostics.candidate_scales[1],
                nht_scene_units_per_metre=1.2,
            ),
        ),
        maximum_relative_scale_deviation=0.2,
        alignment_trace=drifted_trace,
    )
    with pytest.raises(ValueError, match="Native candidate scale deviation"):
        replace(evidence, diagnostics=diagnostics)


def test_duplicate_overlap_is_rejected_after_final_kabsch_refit(
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    first, second = alignment_evidence.candidates
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = (4.0, 2.0, 0.0)
    duplicate_transform = RigidTransform.from_matrix(matrix)
    duplicate = CandidateEvidence(
        court_instance_id=second.court_instance_id,
        candidate_id=second.candidate_id,
        fit=CorrespondenceSet(
            points_court=second.fit.points_court,
            points_scene=duplicate_transform.apply(second.fit.points_court),
            camera_ids=second.fit.camera_ids,
        ),
        holdout=CorrespondenceSet(
            points_court=second.holdout.points_court,
            points_scene=duplicate_transform.apply(second.holdout.points_court),
            camera_ids=second.holdout.camera_ids,
        ),
    )
    overlapping = replace(
        alignment_evidence,
        candidates=(first, duplicate),
    )
    evidence = _with_whole_court_evidence(overlapping)

    with pytest.raises(ValueError, match="footprint_overlap_fraction"):
        fit_alignment(evidence, policy=alignment_policy)


def test_footprint_overlap_is_independent_of_court_normal_winding() -> None:
    first = RigidTransform.from_matrix(np.eye(4, dtype=np.float64))
    opposite_normal = np.diag((1.0, -1.0, -1.0, 1.0))
    second = RigidTransform.from_matrix(opposite_normal)

    (metrics,) = evaluate_court_topology(
        (("candidate-a", first), ("candidate-b", second))
    )

    assert metrics.center_separation_metres == pytest.approx(0.0)
    assert metrics.footprint_overlap_fraction == pytest.approx(1.0)


def _whole_court_settings() -> WholeCourtEvidenceSettings:
    return WholeCourtEvidenceSettings(
        required_court_count=2,
        maximum_common_scale_relative_deviation=0.07290400972053463,
        maximum_center_refit_displacement_metres=(
            0.07290400972053463 * np.hypot(HALF_DOUBLES_WIDTH, HALF_LENGTH) + 0.3
        ),
        minimum_distinct_offset_levels=2,
        minimum_matches_per_offset_level=3,
        minimum_level_camera_count=2,
        minimum_secondary_tangential_span_metres=0.6,
        minimum_longitudinal_offset_span_metres=8.23,
        minimum_longitudinal_tangential_span_metres=12.8,
        minimum_transverse_offset_span_metres=12.8,
        minimum_transverse_tangential_span_metres=8.23,
        samples_per_metre=3.0,
        inlier_distance_metres=0.3,
        minimum_inlier_fraction=0.9,
        maximum_q95_error_metres=0.1,
        minimum_semantic_segment_inlier_fraction=0.8,
        minimum_center_separation_metres=10.97,
        maximum_footprint_overlap_fraction=1.0e-9,
    )


def _with_whole_court_evidence(
    evidence: AlignmentEvidence,
    *,
    fit_points_court: np.ndarray | None = None,
    holdout_points_court: np.ndarray | None = None,
    diagnostic_segment_names: set[str] | None = None,
) -> AlignmentEvidence:
    settings = _whole_court_settings()
    fit_points = (
        _identifiable_points()
        if fit_points_court is None
        else np.asarray(fit_points_court, dtype=np.float64)
    )
    holdout_points = (
        fit_points
        if holdout_points_court is None
        else np.asarray(holdout_points_court, dtype=np.float64)
    )
    transforms = [
        fit_rigid_transform(candidate.fit) for candidate in evidence.candidates
    ]
    candidates = tuple(
        CandidateEvidence(
            court_instance_id=candidate.court_instance_id,
            candidate_id=candidate.candidate_id,
            fit=_transformed_correspondences(
                fit_points,
                transform=transform,
                camera_ids=evidence.partitions.fit_camera_ids,
            ),
            holdout=_transformed_correspondences(
                holdout_points,
                transform=transform,
                camera_ids=evidence.partitions.holdout_camera_ids,
            ),
        )
        for candidate, transform in zip(evidence.candidates, transforms, strict=True)
    )
    sampled = sample_court_line_segments(settings.samples_per_metre)
    selected = [
        points
        for segment, points in sampled
        if diagnostic_segment_names is None or segment.name in diagnostic_segment_names
    ]
    court_points = np.column_stack(
        (np.concatenate(selected), np.zeros(sum(len(item) for item in selected)))
    )
    scene_parts = [transform.apply(court_points) for transform in transforms]
    measured_metric = np.concatenate(scene_parts)
    measured_nht = evidence.metric_adapter.nht_from_metric_points(measured_metric)
    measured_lines = tuple(
        MeasuredCameraLines(
            camera_id=camera_id,
            points_nht_scene=measured_nht,
        )
        for camera_id in (
            evidence.partitions.fit_camera_ids + evidence.partitions.holdout_camera_ids
        )
    )
    cameras = tuple(
        replace(
            diagnostic,
            projected_line_point_count=len(measured_nht),
        )
        for diagnostic in evidence.diagnostics.cameras
    )
    plane = evidence.ground_plane_frame
    basis_u = np.asarray(plane.basis_u_metric_scene, dtype=np.float64)
    basis_v = np.asarray(plane.basis_v_metric_scene, dtype=np.float64)
    final_step = evidence.alignment_trace.step(AlignmentTracePhase.FINAL_ALIGNMENT)
    updated_final_step = replace(
        final_step,
        candidates=tuple(
            replace(
                state,
                center_uv_metres=cast(
                    tuple[float, float],
                    tuple(
                        float(item)
                        for item in plane.to_uv(
                            transform.matrix()[None, :3, 3]
                        )[0]
                    ),
                ),
                orientation_radians=float(
                    np.arctan2(
                        float(transform.matrix()[:3, 0] @ basis_v),
                        float(transform.matrix()[:3, 0] @ basis_u),
                    )
                ),
            )
            for state, transform in zip(
                final_step.candidates,
                transforms,
                strict=True,
            )
        ),
    )
    trace = replace(
        evidence.alignment_trace,
        steps=(*evidence.alignment_trace.steps[:-1], updated_final_step),
    )
    return replace(
        evidence,
        candidates=candidates,
        measured_camera_lines=measured_lines,
        diagnostics=replace(
            evidence.diagnostics,
            cameras=cameras,
            alignment_trace=trace,
        ),
        whole_court_settings=settings,
    )


def _identifiable_points(
    *,
    longitudinal_offsets: tuple[float, float] = (
        -HALF_DOUBLES_WIDTH,
        HALF_DOUBLES_WIDTH,
    ),
    longitudinal_tangential_limit: float = 10.0,
) -> NDArray[np.float64]:
    points: NDArray[np.float64] = np.asarray(
        np.concatenate(
            (
                _longitudinal_points(
                    offsets=longitudinal_offsets,
                    tangential_limit=longitudinal_tangential_limit,
                ),
                _transverse_points(),
            )
        ),
        dtype=np.float64,
    )
    return points


def _longitudinal_points(
    *,
    offsets: tuple[float, ...],
    tangential_limit: float,
) -> NDArray[np.float64]:
    tangential = np.linspace(-tangential_limit, tangential_limit, 41)
    points: NDArray[np.float64] = np.asarray(
        [(offset, position, 0.0) for offset in offsets for position in tangential],
        dtype=np.float64,
    )
    return points


def _transverse_points() -> NDArray[np.float64]:
    points: NDArray[np.float64] = np.asarray(
        np.concatenate(
            (
                _transverse_level_points(offset=-HALF_LENGTH),
                _transverse_level_points(offset=HALF_LENGTH),
            )
        ),
        dtype=np.float64,
    )
    return points


def _transverse_level_points(*, offset: float) -> NDArray[np.float64]:
    tangential = np.linspace(-4.63, 4.63, 31)
    points: NDArray[np.float64] = np.asarray(
        [(position, offset, 0.0) for position in tangential],
        dtype=np.float64,
    )
    return points


def _repeated_camera_correspondences(points_court: np.ndarray) -> CorrespondenceSet:
    points = np.asarray(points_court, dtype=np.float64)
    repeated = np.concatenate((points, points))
    return CorrespondenceSet(
        points_court=repeated,
        points_scene=repeated,
        camera_ids=("camera-0",) * len(points) + ("camera-1",) * len(points),
    )


def _anchor_secondary_components(
    *,
    secondary_positions: tuple[float, ...] = (-0.9, -0.3, 0.3, 0.9),
) -> tuple[np.ndarray, np.ndarray]:
    longitudinal_anchor = np.asarray(
        [
            (-HALF_DOUBLES_WIDTH, tangential, 0.0)
            for tangential in np.linspace(-7.0, 7.0, 49)
        ],
        dtype=np.float64,
    )
    longitudinal_secondary = np.asarray(
        [(HALF_SINGLES_WIDTH, tangential, 0.0) for tangential in secondary_positions],
        dtype=np.float64,
    )
    transverse_anchor = np.asarray(
        [(tangential, -HALF_LENGTH, 0.0) for tangential in np.linspace(-4.3, 4.3, 37)],
        dtype=np.float64,
    )
    transverse_positions = tuple(
        position for position in secondary_positions if position != 0.0
    )
    if len(transverse_positions) < 3:
        transverse_positions = secondary_positions
    transverse_secondary = np.asarray(
        [
            (tangential, SERVICE_LINE_DISTANCE, 0.0)
            for tangential in transverse_positions
        ],
        dtype=np.float64,
    )
    return (
        np.concatenate((longitudinal_anchor, transverse_anchor)),
        np.concatenate((longitudinal_secondary, transverse_secondary)),
    )


def _anchor_secondary_points(
    *,
    secondary_positions: tuple[float, ...] = (-0.9, -0.3, 0.3, 0.9),
) -> NDArray[np.float64]:
    anchor, secondary = _anchor_secondary_components(
        secondary_positions=secondary_positions
    )
    points: NDArray[np.float64] = np.asarray(
        np.concatenate((anchor, secondary)), dtype=np.float64
    )
    return points


def _camera_repeated_correspondences(
    points_court: np.ndarray,
    *,
    camera_count: int,
) -> CorrespondenceSet:
    return _partitioned_correspondences(
        [np.asarray(points_court, dtype=np.float64) for _index in range(camera_count)]
    )


def _partitioned_correspondences(parts: list[np.ndarray]) -> CorrespondenceSet:
    points = np.concatenate(parts)
    camera_ids = tuple(
        f"camera-{index}" for index, part in enumerate(parts) for _point in part
    )
    return CorrespondenceSet(
        points_court=points,
        points_scene=points,
        camera_ids=camera_ids,
    )


def _identifiability_payload(
    correspondences: CorrespondenceSet,
    *,
    minimum_camera_count: int = 2,
) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        evaluate_court_identifiability(
            correspondences,
            minimum_camera_count=minimum_camera_count,
            settings=_whole_court_settings(),
        ).to_dict(
            minimum_camera_count=minimum_camera_count,
            settings=_whole_court_settings(),
        ),
    )


def _transformed_correspondences(
    points_court: np.ndarray,
    *,
    transform: RigidTransform,
    camera_ids: tuple[str, ...],
) -> CorrespondenceSet:
    repeated = np.concatenate([points_court for _camera_id in camera_ids])
    repeated_camera_ids = tuple(
        camera_id for camera_id in camera_ids for _point in points_court
    )
    return CorrespondenceSet(
        points_court=repeated,
        points_scene=transform.apply(repeated),
        camera_ids=repeated_camera_ids,
    )
