from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.synthetic_data_generation.dataset.court.schema import (
    CourtDatasetSchemaVersion,
)
from src.synthetic_data_generation.dataset.court.semantic_pre_render import (
    INSUFFICIENT_PRE_RENDER_SEMANTIC_COVERAGE_REASON,
    CourtSemanticFrameDisposition,
    CourtSemanticPreRenderDecision,
    court_semantic_phase_disposition_digest,
    evaluate_court_semantic_pre_render,
)
from src.synthetic_data_generation.scene_contract import (
    CourtInstance,
    MultiCourtLayout,
    RigidTransform,
    SceneCamera,
)


def test_semantic_pre_render_accepts_coverage_and_preserves_projection() -> None:
    layout = _layout()
    camera = _camera(camera_id="sample-a", center=(0.0, -30.0, 12.0))

    decision = evaluate_court_semantic_pre_render(
        camera,
        layout,
        schema_version=CourtDatasetSchemaVersion.V4,
    )

    assert decision.accepted
    assert decision.disposition == "accepted"
    assert decision.rejection_reasons == ()
    assert decision.projection is not None
    assert decision.projection.camera_id == camera.camera_id


def test_semantic_pre_render_catches_ambiguity_with_the_exact_reason() -> None:
    layout = _layout()
    camera = _camera(camera_id="sample-a", center=(30.0, 0.0, 12.0))

    decision = evaluate_court_semantic_pre_render(
        camera,
        layout,
        schema_version=CourtDatasetSchemaVersion.V4,
    )

    assert not decision.accepted
    assert decision.projection is None
    assert decision.rejection_reasons == ("ambiguous_camera_relative_near_far:court-a",)


def test_semantic_pre_render_rejects_fewer_than_four_in_frame_points() -> None:
    layout = _layout()
    camera = SceneCamera(
        camera_id="sample-a",
        source_frame_index=7,
        width=64,
        height=48,
        intrinsics=(100.0, 0.0, 31.5, 0.0, 100.0, 23.5, 0.0, 0.0, 1.0),
        camera_to_scene=RigidTransform.identity(),
        image_path="generated/sample-a.png",
    )

    decision = evaluate_court_semantic_pre_render(
        camera,
        layout,
        schema_version=CourtDatasetSchemaVersion.V1,
    )

    assert not decision.accepted
    assert decision.projection is not None
    assert decision.rejection_reasons == (
        INSUFFICIENT_PRE_RENDER_SEMANTIC_COVERAGE_REASON,
    )


def test_phase_digest_ignores_incidental_ids_and_input_order() -> None:
    layout = _layout()
    first = _frame(
        frame_index=0,
        camera=_camera(camera_id="sample-a", center=(0.0, -30.0, 12.0)),
        layout=layout,
    )
    second = _frame(
        frame_index=1,
        camera=_camera(camera_id="sample-b", center=(5.0, -30.0, 12.0)),
        layout=layout,
    )
    renamed = tuple(_rename_frame(item, suffix="renamed") for item in (first, second))

    expected = _digest((first, second))

    assert _digest((second, first)) == expected
    assert _digest(tuple(reversed(renamed))) == expected


def test_phase_digest_binds_geometry_frame_semantics_and_phase_authority() -> None:
    layout = _layout()
    first = _frame(
        frame_index=0,
        camera=_camera(camera_id="sample-a", center=(0.0, -30.0, 12.0)),
        layout=layout,
    )
    second = _frame(
        frame_index=1,
        camera=_camera(camera_id="sample-b", center=(5.0, -30.0, 12.0)),
        layout=layout,
    )
    dispositions = (first, second)
    baseline = _digest(dispositions)

    changed_intrinsics_camera = replace(
        second.camera,
        intrinsics=(510.0, 0.0, 319.5, 0.0, 500.0, 239.5, 0.0, 0.0, 1.0),
    )
    changed_intrinsics = _frame(
        frame_index=1,
        camera=changed_intrinsics_camera,
        layout=layout,
    )
    changed_frame = replace(second, trajectory_frame_index=2)
    rejected_decision = CourtSemanticPreRenderDecision(
        camera_id=second.camera.camera_id,
        projection=second.decision.projection,
        rejection_reasons=(INSUFFICIENT_PRE_RENDER_SEMANTIC_COVERAGE_REASON,),
    )
    changed_semantics = replace(second, decision=rejected_decision)

    assert _digest((first, changed_intrinsics)) != baseline
    assert _digest((first, changed_frame)) != baseline
    assert _digest((first, changed_semantics)) != baseline
    assert (
        _digest(dispositions, schema_version=CourtDatasetSchemaVersion.V3) != baseline
    )
    assert _digest(dispositions, trajectory_group_id="group-b") != baseline
    assert _digest(dispositions, phase_index=1, phase_count=2) != baseline
    assert _digest(dispositions, phase_index=0, phase_count=2) != baseline


def _layout() -> MultiCourtLayout:
    transform = RigidTransform.identity()
    court = CourtInstance(
        court_instance_id="court-a",
        candidate_id="candidate-a",
        scene_from_court=transform,
        court_from_scene=transform,
        fit_status="accepted",
        fit_metrics={"rms_error_m": 0.01},
        holdout_status="accepted",
        holdout_metrics={"rms_error_m": 0.02},
    )
    return MultiCourtLayout(
        courts=(court,),
        complex_bounds_scene=(-40.0, -40.0, -1.0, 40.0, 40.0, 20.0),
        primary_court_instance_id=court.court_instance_id,
    )


def _camera(
    *,
    camera_id: str,
    center: tuple[float, float, float],
) -> SceneCamera:
    center_array = np.asarray(center, dtype=np.float64)
    forward = -center_array
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.column_stack((right, down, forward))
    matrix[:3, 3] = center_array
    return SceneCamera(
        camera_id=camera_id,
        source_frame_index=42,
        width=640,
        height=480,
        intrinsics=(500.0, 0.0, 319.5, 0.0, 500.0, 239.5, 0.0, 0.0, 1.0),
        camera_to_scene=RigidTransform.from_matrix(matrix),
        image_path=f"generated/{camera_id}.png",
    )


def _frame(
    *,
    frame_index: int,
    camera: SceneCamera,
    layout: MultiCourtLayout,
) -> CourtSemanticFrameDisposition:
    return CourtSemanticFrameDisposition(
        trajectory_frame_index=frame_index,
        camera=camera,
        decision=evaluate_court_semantic_pre_render(
            camera,
            layout,
            schema_version=CourtDatasetSchemaVersion.V4,
        ),
    )


def _rename_frame(
    disposition: CourtSemanticFrameDisposition,
    *,
    suffix: str,
) -> CourtSemanticFrameDisposition:
    camera = replace(
        disposition.camera,
        camera_id=f"{disposition.camera.camera_id}-{suffix}",
        source_frame_index=999,
        image_path=f"generated/{suffix}.png",
    )
    projection = disposition.decision.projection
    assert projection is not None
    decision = replace(
        disposition.decision,
        camera_id=camera.camera_id,
        projection=replace(projection, camera_id=camera.camera_id),
    )
    return replace(disposition, camera=camera, decision=decision)


def _digest(
    dispositions: tuple[CourtSemanticFrameDisposition, ...],
    *,
    schema_version: CourtDatasetSchemaVersion = CourtDatasetSchemaVersion.V4,
    trajectory_group_id: str = "group-a",
    phase_index: int = 0,
    phase_count: int = 1,
) -> str:
    return court_semantic_phase_disposition_digest(
        dispositions,
        schema_version=schema_version,
        trajectory_group_id=trajectory_group_id,
        phase_index=phase_index,
        phase_count=phase_count,
    )
