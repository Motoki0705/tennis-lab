"""Pure per-camera target-court resolution for Court dataset v2."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.contracts import TargetCourtBinding
from src.synthetic_data_generation.dataset.court.contracts import (
    OrbitCenterKind,
    OrbitTrajectorySpec,
    ResolvedTargetCourtV2,
    TargetCourtPolicyV2,
    TargetCourtResolutionPolicy,
)
from src.synthetic_data_generation.scene_contract import MultiCourtLayout, SceneCamera

NEAREST_COURT_TIE_TOLERANCE_M = 1.0e-9
CAMERA_FORWARD_AXIS_ATOL = 1.0e-9


def target_court_policy_for_trajectory(
    trajectory: OrbitTrajectorySpec,
) -> TargetCourtPolicyV2:
    """Create the only valid v2 target policy for one trajectory centre."""
    if not isinstance(trajectory, OrbitTrajectorySpec):
        raise TypeError("trajectory must be an OrbitTrajectorySpec.")
    if trajectory.center_kind is OrbitCenterKind.COURT:
        return TargetCourtPolicyV2(
            mode=TargetCourtResolutionPolicy.TRAJECTORY_CENTER_COURT,
            centre_court_instance_id=trajectory.center_court_instance_id,
        )
    return TargetCourtPolicyV2(
        mode=TargetCourtResolutionPolicy.NEAREST_CAMERA,
        centre_court_instance_id=None,
    )


def resolve_target_court(
    *,
    policy: TargetCourtPolicyV2,
    camera_center_scene_m: Sequence[float],
    layout: MultiCourtLayout,
    selection_seed: int,
) -> ResolvedTargetCourtV2:
    """Resolve one v2 sample target using 3-D metric scene distance.

    Court-centred policies are fixed by contract.  Nearest-camera policies
    treat distances within :data:`NEAREST_COURT_TIE_TOLERANCE_M` as tied and
    select the lexicographically smallest court ID.
    """
    if not isinstance(policy, TargetCourtPolicyV2):
        raise TypeError("policy must be a TargetCourtPolicyV2.")
    if not isinstance(layout, MultiCourtLayout):
        raise TypeError("layout must be a MultiCourtLayout.")
    if isinstance(selection_seed, bool) or not isinstance(selection_seed, int):
        raise TypeError("selection_seed must be an integer.")
    camera_center = _scene_point(camera_center_scene_m, name="camera_center_scene_m")
    distances = {
        court.court_instance_id: _court_center_distance(
            camera_center,
            court.scene_from_court.apply,
        )
        for court in layout.courts
    }
    if policy.mode is TargetCourtResolutionPolicy.TRAJECTORY_CENTER_COURT:
        assert policy.centre_court_instance_id is not None
        try:
            selected_court = layout.court(policy.centre_court_instance_id)
        except KeyError as error:
            raise ValueError(
                "trajectory_center_court policy references an unaccepted court."
            ) from error
    else:
        tied_ids = nearest_court_tie_ids(
            camera_center_scene_m=tuple(float(value) for value in camera_center),
            layout=layout,
        )
        selected_court = layout.court(tied_ids[0])
    selected_distance = distances[selected_court.court_instance_id]
    return ResolvedTargetCourtV2(
        binding=TargetCourtBinding(
            court_instance_id=selected_court.court_instance_id,
            candidate_id=selected_court.candidate_id,
            scene_from_court=selected_court.scene_from_court,
            selection_seed=selection_seed,
        ),
        resolution_policy=policy.mode,
        camera_to_court_center_distance_m=selected_distance,
    )


def nearest_court_tie_ids(
    *,
    camera_center_scene_m: Sequence[float],
    layout: MultiCourtLayout,
) -> tuple[str, ...]:
    """Return every equally-nearest court ID in lexical order for diagnostics."""
    if not isinstance(layout, MultiCourtLayout):
        raise TypeError("layout must be a MultiCourtLayout.")
    camera_center = _scene_point(camera_center_scene_m, name="camera_center_scene_m")
    distances = {
        court.court_instance_id: _court_center_distance(
            camera_center,
            court.scene_from_court.apply,
        )
        for court in layout.courts
    }
    minimum_distance = min(distances.values())
    return tuple(
        sorted(
            court_id
            for court_id, distance in distances.items()
            if distance <= minimum_distance + NEAREST_COURT_TIE_TOLERANCE_M
        )
    )


def resolved_court_look_at_scene(
    *,
    target_court: ResolvedTargetCourtV2,
    layout: MultiCourtLayout,
    look_at_height_m: float,
) -> NDArray[np.float64]:
    """Transform local ``(0, 0, height)`` through the resolved court."""
    if not isinstance(target_court, ResolvedTargetCourtV2):
        raise TypeError("target_court must be a ResolvedTargetCourtV2.")
    if isinstance(look_at_height_m, bool) or not isinstance(
        look_at_height_m, (int, float)
    ):
        raise TypeError("look_at_height_m must be numeric.")
    height = float(look_at_height_m)
    if not math.isfinite(height) or height < 0.0:
        raise ValueError("look_at_height_m must be finite and non-negative.")
    court = layout.court(target_court.binding.court_instance_id)
    _require_binding_matches_layout(target_court.binding, layout=layout)
    transformed = court.scene_from_court.apply(
        np.asarray(((0.0, 0.0, height),), dtype=np.float64)
    )[0]
    return np.asarray(transformed, dtype=np.float64)


def validate_resolved_target_court(
    *,
    policy: TargetCourtPolicyV2,
    camera_center_scene_m: Sequence[float],
    target_court: ResolvedTargetCourtV2,
    layout: MultiCourtLayout,
) -> None:
    """Recompute v2 geometry and reject stale or fabricated target evidence."""
    expected = resolve_target_court(
        policy=policy,
        camera_center_scene_m=camera_center_scene_m,
        layout=layout,
        selection_seed=target_court.binding.selection_seed,
    )
    if target_court.resolution_policy is not expected.resolution_policy:
        raise ValueError("Stored target resolution policy is incorrect.")
    if target_court.binding.to_dict() != expected.binding.to_dict():
        raise ValueError("Stored target court binding is incorrect.")
    if not math.isclose(
        target_court.camera_to_court_center_distance_m,
        expected.camera_to_court_center_distance_m,
        abs_tol=NEAREST_COURT_TIE_TOLERANCE_M,
        rel_tol=0.0,
    ):
        raise ValueError("Stored camera-to-court distance is incorrect.")


def validate_camera_looks_at_resolved_court(
    *,
    camera: SceneCamera,
    target_court: ResolvedTargetCourtV2,
    layout: MultiCourtLayout,
    look_at_height_m: float,
    atol: float = CAMERA_FORWARD_AXIS_ATOL,
) -> None:
    """Require OpenCV local +Z to point at the resolved local court target."""
    target_scene = resolved_court_look_at_scene(
        target_court=target_court,
        layout=layout,
        look_at_height_m=look_at_height_m,
    )
    _validate_camera_forward_axis(
        camera=camera,
        target_scene=target_scene,
        atol=atol,
    )


def validate_camera_looks_at_resolved_binding(
    *,
    camera: SceneCamera,
    target_court: ResolvedTargetCourtV2,
    look_at_height_m: float,
    atol: float = CAMERA_FORWARD_AXIS_ATOL,
) -> None:
    """Validate a persisted camera against its sample-owned court binding."""
    if not isinstance(target_court, ResolvedTargetCourtV2):
        raise TypeError("target_court must be a ResolvedTargetCourtV2.")
    if isinstance(look_at_height_m, bool) or not isinstance(
        look_at_height_m, (int, float)
    ):
        raise TypeError("look_at_height_m must be numeric.")
    height = float(look_at_height_m)
    if not math.isfinite(height) or height < 0.0:
        raise ValueError("look_at_height_m must be finite and non-negative.")
    target_scene = target_court.binding.scene_from_court.apply(
        np.asarray(((0.0, 0.0, height),), dtype=np.float64)
    )[0]
    _validate_camera_forward_axis(
        camera=camera,
        target_scene=np.asarray(target_scene, dtype=np.float64),
        atol=atol,
    )


def _validate_camera_forward_axis(
    *,
    camera: SceneCamera,
    target_scene: NDArray[np.float64],
    atol: float,
) -> None:
    """Require one camera's local +Z to point at an exact scene target."""
    if not isinstance(camera, SceneCamera):
        raise TypeError("camera must be a SceneCamera.")
    if not math.isfinite(atol) or atol <= 0.0:
        raise ValueError("atol must be positive and finite.")
    matrix = camera.camera_to_scene.matrix()
    direction = target_scene - matrix[:3, 3]
    norm = float(np.linalg.norm(direction))
    if norm <= NEAREST_COURT_TIE_TOLERANCE_M:
        raise ValueError("Camera centre and resolved look-at target must differ.")
    expected_forward = direction / norm
    if not np.allclose(matrix[:3, 2], expected_forward, atol=atol, rtol=0.0):
        raise ValueError(
            "Camera forward axis misses the resolved court look-at target."
        )


def _court_center_distance(
    camera_center: NDArray[np.float64],
    transform: object,
) -> float:
    if not callable(transform):
        raise TypeError("court transform must be callable.")
    court_center = transform(np.zeros((1, 3), dtype=np.float64))[0]
    return float(np.linalg.norm(camera_center - court_center))


def _scene_point(value: Sequence[float], *, name: str) -> NDArray[np.float64]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a numeric three-vector.")
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite numeric three-vector.")
    return result


def _require_binding_matches_layout(
    binding: TargetCourtBinding,
    *,
    layout: MultiCourtLayout,
) -> None:
    court = layout.court(binding.court_instance_id)
    if binding.candidate_id != court.candidate_id or not np.allclose(
        binding.scene_from_court.matrix(),
        court.scene_from_court.matrix(),
        atol=1.0e-9,
        rtol=0.0,
    ):
        raise ValueError("Resolved target binding disagrees with the accepted layout.")


__all__ = [
    "CAMERA_FORWARD_AXIS_ATOL",
    "NEAREST_COURT_TIE_TOLERANCE_M",
    "nearest_court_tie_ids",
    "resolve_target_court",
    "resolved_court_look_at_scene",
    "target_court_policy_for_trajectory",
    "validate_camera_looks_at_resolved_binding",
    "validate_camera_looks_at_resolved_court",
    "validate_resolved_target_court",
]
