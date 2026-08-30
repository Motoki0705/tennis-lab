from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from src.synthetic_data_generation.alignment.contracts import (
    AlignmentAcceptancePolicy,
    AlignmentPartitions,
    AlignmentResult,
    CandidateAlignment,
    MetricSceneAdapter,
    PartitionAssessment,
    PartitionMetrics,
    PartitionThresholds,
)
from src.synthetic_data_generation.dataset.contracts import TargetCourtBinding
from src.synthetic_data_generation.dataset.court.components.camera_sampling.selection import (
    _metric_cameras_from_nht_export,
)
from src.synthetic_data_generation.dataset.court.contracts import CourtDatasetPlan
from src.synthetic_data_generation.dataset.court.rendering.nht import (
    _nht_camera_from_metric_plan,
    _validate_plan_alignment,
    validate_pre_render_plan,
)
from src.synthetic_data_generation.scene_contract import (
    MultiCourtLayout,
    RigidTransform,
    SceneCamera,
)


def test_captured_and_planned_cameras_round_trip_at_nht_boundary() -> None:
    alignment = _alignment()
    adapter = alignment.metric_adapter
    metric_camera = _camera(_rigid(angle=0.35, translation=(24.0, -8.0, 6.0)))
    nht_pose = adapter.nht_from_metric_camera(metric_camera.camera_to_scene)
    exported_camera = _camera(nht_pose)

    converted = _metric_cameras_from_nht_export(
        (exported_camera,),
        metric_adapter=adapter,
    )
    assert len(converted) == 1
    assert converted[0].camera_id == exported_camera.camera_id
    assert converted[0].intrinsics == exported_camera.intrinsics
    assert np.allclose(
        converted[0].camera_to_scene.matrix(),
        metric_camera.camera_to_scene.matrix(),
        atol=1.0e-9,
        rtol=0.0,
    )
    request_camera = _nht_camera_from_metric_plan(
        converted[0],
        alignment=alignment,
    )
    assert request_camera.camera_id == exported_camera.camera_id
    assert request_camera.intrinsics == exported_camera.intrinsics
    assert np.allclose(
        request_camera.camera_to_scene.matrix(),
        exported_camera.camera_to_scene.matrix(),
        atol=1.0e-9,
        rtol=0.0,
    )
    assert np.allclose(
        converted[0].camera_to_scene.matrix(),
        metric_camera.camera_to_scene.matrix(),
        atol=1.0e-9,
        rtol=0.0,
    )


def test_camera_translation_ratio_is_the_public_similarity_reciprocal() -> None:
    adapter = _alignment().metric_adapter
    nht_points = np.asarray(((0.0, 0.0, 0.0), (1.0, -2.0, 3.0)))
    metric_points = adapter.metric_from_nht_points(nht_points)
    nht_distance = float(np.linalg.norm(nht_points[1] - nht_points[0]))
    metric_distance = float(np.linalg.norm(metric_points[1] - metric_points[0]))

    assert metric_distance / nht_distance == pytest.approx(
        1.0 / adapter.nht_scene_units_per_metre
    )

    nht_camera = _camera(_rigid(angle=0.2, translation=(1.0, -2.0, 3.0)))
    metric_camera = adapter.metric_from_nht_camera(nht_camera.camera_to_scene)
    nht_origin_metric = adapter.metric_from_nht_points(
        np.zeros((1, 3), dtype=np.float64)
    )[0]
    camera_translation_metric = metric_camera.matrix()[:3, 3] - nht_origin_metric
    assert np.linalg.norm(camera_translation_metric) / np.linalg.norm(
        nht_camera.camera_to_scene.matrix()[:3, 3]
    ) == pytest.approx(1.0 / adapter.nht_scene_units_per_metre)


def test_boundary_requires_complete_matching_alignment_inventory() -> None:
    alignment = _alignment()
    camera = _camera(_rigid(angle=0.0, translation=(20.0, 0.0, 5.0)))
    with pytest.raises(ValueError, match="requires captured"):
        _metric_cameras_from_nht_export(
            (),
            metric_adapter=alignment.metric_adapter,
        )
    with pytest.raises(TypeError, match="complete AlignmentResult"):
        _nht_camera_from_metric_plan(
            camera,
            alignment=cast(AlignmentResult, alignment.metric_adapter),
        )

    court = alignment.layout.courts[0]
    mismatched_binding = TargetCourtBinding(
        court_instance_id=court.court_instance_id,
        candidate_id="another-candidate",
        scene_from_court=court.scene_from_court,
        selection_seed=17,
    )
    plan = cast(
        CourtDatasetPlan,
        SimpleNamespace(
            groups=(SimpleNamespace(target_court=mismatched_binding),),
        ),
    )
    with pytest.raises(ValueError, match="complete alignment inventory"):
        _validate_plan_alignment(plan, alignment)


def test_pre_render_gate_classifies_camera_without_semantic_coverage() -> None:
    alignment = _alignment()
    camera = _camera(RigidTransform.identity())
    sample = SimpleNamespace(sample_id="camera-a", camera=camera)
    plan = cast(
        CourtDatasetPlan,
        SimpleNamespace(
            groups=(),
            proposal_count=1,
            policy=SimpleNamespace(proposal_budget=1),
            samples=(sample,),
        ),
    )

    evaluation = validate_pre_render_plan(plan, alignment=alignment)

    assert tuple(item.camera_id for item in evaluation.projections) == ("camera-a",)
    assert evaluation.rejected_sample_ids == ("camera-a",)


def _alignment() -> AlignmentResult:
    thresholds = PartitionThresholds(
        minimum_camera_count=1,
        minimum_correspondence_count=3,
        inlier_distance_m=0.01,
        minimum_inlier_fraction=1.0,
        maximum_rms_error_m=0.01,
        maximum_q95_error_m=0.01,
    )
    policy = AlignmentAcceptancePolicy(fit=thresholds, holdout=thresholds)
    fit = PartitionAssessment.evaluate(_metrics("fit-camera"), thresholds)
    holdout = PartitionAssessment.evaluate(_metrics("holdout-camera"), thresholds)
    scene_from_court = _rigid(angle=0.1, translation=(2.0, 3.0, 0.5))
    candidate = CandidateAlignment(
        court_instance_id="court-a",
        candidate_id="candidate-a",
        scene_from_court=scene_from_court,
        court_from_scene=scene_from_court.inverse(),
        fit=fit,
        holdout=holdout,
    )
    layout = MultiCourtLayout(
        courts=(candidate.to_court_instance(),),
        complex_bounds_scene=(-12.0, -15.0, -1.0, 18.0, 20.0, 10.0),
        primary_court_instance_id="court-a",
    )
    similarity = np.eye(4, dtype=np.float64)
    angle = -0.2
    scale = 0.25
    similarity[:2, :2] = scale * np.asarray(
        (
            (np.cos(angle), -np.sin(angle)),
            (np.sin(angle), np.cos(angle)),
        ),
        dtype=np.float64,
    )
    similarity[2, 2] = scale
    similarity[:3, 3] = (0.5, -0.25, 1.0)
    return AlignmentResult(
        partitions=AlignmentPartitions(
            fit_camera_ids=("fit-camera",),
            holdout_camera_ids=("holdout-camera",),
        ),
        policy=policy,
        candidates=(candidate,),
        layout=layout,
        metric_adapter=MetricSceneAdapter.from_nht_scene_from_metric_scene(similarity),
    )


def _metrics(camera_id: str) -> PartitionMetrics:
    return PartitionMetrics(
        camera_ids=(camera_id,),
        correspondence_count=3,
        inlier_count=3,
        inlier_fraction=1.0,
        rms_error_m=0.0,
        q95_error_m=0.0,
        maximum_error_m=0.0,
    )


def _rigid(*, angle: float, translation: tuple[float, float, float]) -> RigidTransform:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:2, :2] = np.asarray(
        (
            (np.cos(angle), -np.sin(angle)),
            (np.sin(angle), np.cos(angle)),
        ),
        dtype=np.float64,
    )
    matrix[:3, 3] = translation
    return RigidTransform.from_matrix(matrix)


def _camera(camera_to_scene: RigidTransform) -> SceneCamera:
    return SceneCamera(
        camera_id="camera-a",
        source_frame_index=3,
        width=64,
        height=48,
        intrinsics=(100.0, 0.0, 31.5, 0.0, 100.0, 23.5, 0.0, 0.0, 1.0),
        camera_to_scene=camera_to_scene,
        image_path="images/camera-a.png",
    )
