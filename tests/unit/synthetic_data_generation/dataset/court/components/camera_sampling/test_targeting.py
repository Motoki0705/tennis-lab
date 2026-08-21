"""Pure geometric tests for v2 per-camera target-court resolution."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.court.components.camera_sampling.targeting import (
    NEAREST_COURT_TIE_TOLERANCE_M,
    nearest_court_tie_ids,
    resolve_target_court,
    target_court_policy_for_trajectory,
    validate_camera_looks_at_resolved_court,
    validate_resolved_target_court,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    OrbitCenterKind,
    OrbitCurveMode,
    OrbitShape,
    OrbitTrajectorySpec,
    TargetCourtPolicyV2,
    TargetCourtResolutionPolicy,
)
from src.synthetic_data_generation.scene_contract import (
    CourtInstance,
    MultiCourtLayout,
    RigidTransform,
    SceneCamera,
)


def _transform(x: float, y: float = 0.0, z: float = 0.0) -> RigidTransform:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = (x, y, z)
    return RigidTransform.from_matrix(matrix)


def _layout(
    *court_specs: tuple[str, tuple[float, float, float]],
) -> MultiCourtLayout:
    courts = []
    for court_id, (x, y, z) in court_specs:
        scene_from_court = _transform(x, y, z)
        courts.append(
            CourtInstance(
                court_instance_id=court_id,
                candidate_id=f"candidate-{court_id}",
                scene_from_court=scene_from_court,
                court_from_scene=scene_from_court.inverse(),
                fit_status="accepted",
                fit_metrics={"rms_error_m": 0.01},
                holdout_status="accepted",
                holdout_metrics={"rms_error_m": 0.02},
            )
        )
    return MultiCourtLayout(
        courts=tuple(courts),
        complex_bounds_scene=(-50.0, -50.0, -5.0, 50.0, 50.0, 110.0),
        primary_court_instance_id=courts[0].court_instance_id,
    )


def _trajectory(
    *, center_kind: OrbitCenterKind, center_court_id: str | None
) -> OrbitTrajectorySpec:
    return OrbitTrajectorySpec(
        trajectory_id=f"trajectory-{center_kind.value}",
        trajectory_group_id=f"group-{center_kind.value}",
        shape=OrbitShape.CIRCLE,
        center_kind=center_kind,
        center_court_instance_id=center_court_id,
        base_radius_m=20.0,
        radius_scale=1.0,
        axis_ratio=1.0,
        orientation_radians=0.0,
        base_height_m=6.0,
        vertical_amplitude_m=0.0,
        vertical_cycles=0,
        vertical_phase_radians=0.0,
        curve_mode=OrbitCurveMode.PLANAR,
    )


def _look_at_camera(
    center: tuple[float, float, float], target: NDArray[np.float64]
) -> SceneCamera:
    center_array = np.asarray(center, dtype=np.float64)
    forward = target - center_array
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0), dtype=np.float64))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.column_stack((right, down, forward))
    matrix[:3, 3] = center_array
    return SceneCamera(
        camera_id="camera-a",
        source_frame_index=0,
        width=64,
        height=48,
        intrinsics=(100.0, 0.0, 31.5, 0.0, 100.0, 23.5, 0.0, 0.0, 1.0),
        camera_to_scene=RigidTransform.from_matrix(matrix),
        image_path="generated/camera-a.png",
    )


def test_court_centred_policy_remains_fixed_when_another_court_is_closer() -> None:
    layout = _layout(
        ("court-fixed", (-10.0, 0.0, 0.0)), ("court-near", (10.0, 0.0, 0.0))
    )
    policy = target_court_policy_for_trajectory(
        _trajectory(
            center_kind=OrbitCenterKind.COURT,
            center_court_id="court-fixed",
        )
    )

    resolved = resolve_target_court(
        policy=policy,
        camera_center_scene_m=(9.0, 0.0, 0.0),
        layout=layout,
        selection_seed=695,
    )

    assert policy.mode is TargetCourtResolutionPolicy.TRAJECTORY_CENTER_COURT
    assert resolved.binding.court_instance_id == "court-fixed"
    assert resolved.camera_to_court_center_distance_m == pytest.approx(19.0)


def test_nearest_policy_switches_by_sample_and_uses_lexical_tie_break() -> None:
    layout = _layout(("court-z", (-10.0, 0.0, 0.0)), ("court-a", (10.0, 0.0, 0.0)))
    policy = target_court_policy_for_trajectory(
        _trajectory(center_kind=OrbitCenterKind.COMPLEX, center_court_id=None)
    )

    left = resolve_target_court(
        policy=policy,
        camera_center_scene_m=(-9.0, 0.0, 0.0),
        layout=layout,
        selection_seed=695,
    )
    right = resolve_target_court(
        policy=policy,
        camera_center_scene_m=(9.0, 0.0, 0.0),
        layout=layout,
        selection_seed=695,
    )
    tie = resolve_target_court(
        policy=policy,
        camera_center_scene_m=(0.0, 0.0, 0.0),
        layout=layout,
        selection_seed=695,
    )

    assert left.binding.court_instance_id == "court-z"
    assert right.binding.court_instance_id == "court-a"
    assert tie.binding.court_instance_id == "court-a"
    assert nearest_court_tie_ids(
        camera_center_scene_m=(0.0, 0.0, 0.0), layout=layout
    ) == ("court-a", "court-z")

    near_tie = nearest_court_tie_ids(
        camera_center_scene_m=(NEAREST_COURT_TIE_TOLERANCE_M / 4.0, 0.0, 0.0),
        layout=layout,
    )
    assert near_tie == ("court-a", "court-z")


def test_nearest_distance_is_metric_three_dimensional_not_planar() -> None:
    layout = _layout(("court-low", (5.0, 0.0, 0.0)), ("court-high", (0.0, 0.0, 100.0)))
    policy = TargetCourtPolicyV2(
        mode=TargetCourtResolutionPolicy.NEAREST_CAMERA,
        centre_court_instance_id=None,
    )

    resolved = resolve_target_court(
        policy=policy,
        camera_center_scene_m=(0.0, 0.0, 99.0),
        layout=layout,
        selection_seed=3,
    )

    assert resolved.binding.court_instance_id == "court-high"
    assert resolved.camera_to_court_center_distance_m == pytest.approx(1.0)


def test_persisted_target_and_forward_axis_are_recomputed_and_rejected_on_mutation() -> (
    None
):
    layout = _layout(("court-a", (4.0, 2.0, 0.5)), ("court-b", (-9.0, 0.0, 0.0)))
    policy = TargetCourtPolicyV2(
        mode=TargetCourtResolutionPolicy.NEAREST_CAMERA,
        centre_court_instance_id=None,
    )
    center = (14.0, -8.0, 7.0)
    resolved = resolve_target_court(
        policy=policy,
        camera_center_scene_m=center,
        layout=layout,
        selection_seed=11,
    )
    target = layout.court(resolved.binding.court_instance_id).scene_from_court.apply(
        np.asarray(((0.0, 0.0, 1.5),), dtype=np.float64)
    )[0]
    camera = _look_at_camera(center, target)

    validate_resolved_target_court(
        policy=policy,
        camera_center_scene_m=center,
        target_court=resolved,
        layout=layout,
    )
    validate_camera_looks_at_resolved_court(
        camera=camera,
        target_court=resolved,
        layout=layout,
        look_at_height_m=1.5,
    )

    with pytest.raises(ValueError, match="distance is incorrect"):
        validate_resolved_target_court(
            policy=policy,
            camera_center_scene_m=center,
            target_court=replace(
                resolved,
                camera_to_court_center_distance_m=(
                    resolved.camera_to_court_center_distance_m + 0.1
                ),
            ),
            layout=layout,
        )
    with pytest.raises(ValueError, match="forward axis misses"):
        validate_camera_looks_at_resolved_court(
            camera=replace(camera, camera_to_scene=_transform(*center)),
            target_court=resolved,
            layout=layout,
            look_at_height_m=1.5,
        )
