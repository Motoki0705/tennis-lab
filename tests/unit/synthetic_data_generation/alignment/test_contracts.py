"""Tests for strict fit/holdout and multi-court alignment contracts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from src.synthetic_data_generation.alignment.contracts import (
    AlignmentAcceptancePolicy,
    AlignmentEvidence,
    AlignmentPartitions,
    AlignmentResult,
    CandidateEvidence,
    CorrespondenceSet,
    MetricSceneAdapter,
)
from src.synthetic_data_generation.alignment.fitting import fit_alignment
from src.synthetic_data_generation.scene_contract import RigidTransform


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


def test_holdout_is_evaluated_without_refitting_and_rejected_court_is_excluded(
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
    )

    result = fit_alignment(evidence, policy=alignment_policy)

    assert result.candidates[0].fit.status.value == "accepted"
    assert result.candidates[0].holdout.status.value == "rejected"
    assert not result.candidates[0].accepted
    assert [court.court_instance_id for court in result.layout.courts] == ["court-b"]
    baseline = fit_alignment(alignment_evidence, policy=alignment_policy)
    np.testing.assert_allclose(
        result.candidates[0].scene_from_court.matrix(),
        baseline.candidates[0].scene_from_court.matrix(),
        atol=1.0e-10,
    )
    expected = fit_alignment(
        AlignmentEvidence(
            partitions=alignment_evidence.partitions,
            candidates=(second,),
            measured_camera_lines=alignment_evidence.measured_camera_lines,
            complex_points_scene=alignment_evidence.complex_points_scene,
            primary_candidate_id="candidate-b",
            metric_adapter=alignment_evidence.metric_adapter,
            diagnostics=type(alignment_evidence.diagnostics)(
                cameras=alignment_evidence.diagnostics.cameras,
                candidate_scales=(alignment_evidence.diagnostics.candidate_scales[1],),
                common_nht_scene_units_per_metre=1.0,
                maximum_relative_scale_deviation=0.0,
            ),
        ),
        policy=alignment_policy,
    )
    np.testing.assert_allclose(
        result.candidates[1].scene_from_court.matrix(),
        expected.candidates[0].scene_from_court.matrix(),
        atol=1.0e-10,
    )


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
    evidence = AlignmentEvidence(
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

    with pytest.raises(ValueError, match="Holdout acceptance failed"):
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
            candidates=(drifted,),
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
