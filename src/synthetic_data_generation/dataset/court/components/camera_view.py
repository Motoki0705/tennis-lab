"""Camera-view canonicalization for Synthetic Court singleton semantics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.synthetic_data_generation.scene_contract import (
    CourtInstance,
    RigidTransform,
    SceneCamera,
)
from src.utils.schema.court import (
    CAMERA_VIEW_HALF_TURN_INDEX,
    NUM_GROUND_COURT_KP,
)

CAMERA_VIEW_MID_PLANE_TOLERANCE_M = 1.0e-6
AMBIGUOUS_CAMERA_RELATIVE_NEAR_FAR_REASON = "ambiguous_camera_relative_near_far"

_IDENTITY_INDEX = tuple(range(NUM_GROUND_COURT_KP))


class AmbiguousCameraRelativeNearFarError(ValueError):
    """Explicit rejection for a camera on one court's local mid-plane."""

    court_instance_id: str
    reason: str

    def __init__(self, court_instance_id: str) -> None:
        self.court_instance_id = court_instance_id
        self.reason = f"{AMBIGUOUS_CAMERA_RELATIVE_NEAR_FAR_REASON}:{court_instance_id}"
        super().__init__(self.reason)


@dataclass(frozen=True, slots=True)
class CameraViewCanonicalization:
    """One side decision shared by camera-view labels and canonical camera pose."""

    semantic_to_physical: tuple[int, ...]
    canonical_from_court: RigidTransform
    camera_from_canonical: RigidTransform
    camera_center_canonical_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.semantic_to_physical not in (
            _IDENTITY_INDEX,
            CAMERA_VIEW_HALF_TURN_INDEX,
        ):
            raise ValueError(
                "Camera-view semantic inventory must be identity or the shared "
                "full half-turn."
            )
        if len(set(self.semantic_to_physical)) != NUM_GROUND_COURT_KP:
            raise ValueError("Camera-view semantic inventory must be a bijection.")
        if not isinstance(self.canonical_from_court, RigidTransform):
            raise TypeError("canonical_from_court must be a RigidTransform.")
        if not isinstance(self.camera_from_canonical, RigidTransform):
            raise TypeError("camera_from_canonical must be a RigidTransform.")
        center = np.asarray(self.camera_center_canonical_m, dtype=np.float64)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("camera_center_canonical_m must be a finite three-vector.")
        object.__setattr__(
            self,
            "camera_center_canonical_m",
            tuple(float(value) for value in center),
        )


def camera_view_canonicalization(
    camera: SceneCamera,
    court: CourtInstance,
) -> CameraViewCanonicalization:
    """Canonicalize one camera/court pair with identity or a proper ``Rz(pi)``."""
    if not isinstance(camera, SceneCamera):
        raise TypeError("camera must be a SceneCamera.")
    if not isinstance(court, CourtInstance):
        raise TypeError("court must be a CourtInstance.")

    camera_center_scene = camera.camera_to_scene.matrix()[:3, 3]
    camera_center_court = court.court_from_scene.apply(
        camera_center_scene.reshape(1, 3)
    )[0]
    local_y = float(camera_center_court[1])
    if abs(local_y) <= CAMERA_VIEW_MID_PLANE_TOLERANCE_M:
        raise AmbiguousCameraRelativeNearFarError(court.court_instance_id)

    canonical_matrix = np.eye(4, dtype=np.float64)
    if local_y < 0.0:
        semantic_to_physical = _IDENTITY_INDEX
    else:
        semantic_to_physical = CAMERA_VIEW_HALF_TURN_INDEX
        canonical_matrix[:3, :3] = np.diag((-1.0, -1.0, 1.0))
    canonical_from_court = RigidTransform.from_matrix(canonical_matrix)

    camera_from_court_matrix = (
        camera.camera_to_scene.inverse().matrix() @ court.scene_from_court.matrix()
    )
    camera_from_canonical = RigidTransform.from_matrix(
        camera_from_court_matrix @ canonical_from_court.inverse().matrix()
    )
    camera_center_canonical = canonical_from_court.apply(
        camera_center_court.reshape(1, 3)
    )[0]
    return CameraViewCanonicalization(
        semantic_to_physical=semantic_to_physical,
        canonical_from_court=canonical_from_court,
        camera_from_canonical=camera_from_canonical,
        camera_center_canonical_m=(
            float(camera_center_canonical[0]),
            float(camera_center_canonical[1]),
            float(camera_center_canonical[2]),
        ),
    )


def validate_finite_camera_view_projection(
    semantic_uv: np.ndarray,
) -> None:
    """Require a finite projected UV coordinate for every V3 singleton class."""
    uv = np.asarray(semantic_uv, dtype=np.float64)
    if uv.shape != (NUM_GROUND_COURT_KP, 2) or not np.isfinite(uv).all():
        raise ValueError(
            "Camera-view projected UV must be a finite (14, 2) array."
        )


__all__ = [
    "AMBIGUOUS_CAMERA_RELATIVE_NEAR_FAR_REASON",
    "CAMERA_VIEW_MID_PLANE_TOLERANCE_M",
    "AmbiguousCameraRelativeNearFarError",
    "CameraViewCanonicalization",
    "camera_view_canonicalization",
    "validate_finite_camera_view_projection",
]
