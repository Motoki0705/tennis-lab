"""Pinhole camera from the scene's court-reference fit, and its Blender pose.

The integrated pipeline stores one approximate pinhole fit per camera in
``metadata["court_reference"]["camera_fits"]`` (OpenCV convention: ``x_cam =
R (X - C)``, +Z forward, +Y image-down). Blender cameras look along their local
-Z with +Y up, so the Blender camera-to-world rotation is ``R^T diag(1,-1,-1)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

_OPENCV_TO_BLENDER_CAMERA = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True, slots=True)
class PinholeCamera:
    """Distortion-free OpenCV pinhole camera in court coordinates."""

    K: NDArray[np.float64]
    R: NDArray[np.float64]
    t: NDArray[np.float64]
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.K.shape != (3, 3) or self.R.shape != (3, 3) or self.t.shape != (3,):
            raise ValueError("PinholeCamera requires K (3,3), R (3,3) and t (3,).")
        if not all(np.isfinite(value).all() for value in (self.K, self.R, self.t)):
            raise ValueError("PinholeCamera matrices must be finite.")
        if min(self.width, self.height) <= 0 or min(self.K[0, 0], self.K[1, 1]) <= 0:
            raise ValueError("Image dimensions and focal lengths must be positive.")
        if not np.allclose(self.K[2], [0.0, 0.0, 1.0]):
            raise ValueError("PinholeCamera.K must have last row [0, 0, 1].")
        if not np.allclose(self.R @ self.R.T, np.eye(3), atol=1e-6):
            raise ValueError("PinholeCamera.R must be orthonormal.")
        if np.linalg.det(self.R) <= 0.0:
            raise ValueError("PinholeCamera.R must be a proper rotation.")

    @property
    def center(self) -> NDArray[np.float64]:
        """Camera center in court coordinates."""
        return -self.R.T @ self.t

    def project(self, points: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """Project ``(..., 3)`` court points to ``(..., 2)`` pixel coordinates."""
        cam = np.asarray(points, dtype=np.float64) @ self.R.T + self.t
        if not np.isfinite(cam).all():
            raise ValueError("Cannot project non-finite points.")
        if np.any(cam[..., 2] <= 0.0):
            raise ValueError("Cannot project points at or behind the camera plane.")
        uv = cam[..., :2] / cam[..., 2:3]
        return uv @ self.K[:2, :2].T + self.K[:2, 2]

    @classmethod
    def from_scene_metadata(
        cls,
        metadata: Mapping[str, Any],
        camera_id: str,
        *,
        width: int,
        height: int,
    ) -> PinholeCamera:
        """Read the court-reference pinhole fit recorded for ``camera_id``."""
        reference = metadata.get("court_reference")
        if not isinstance(reference, Mapping):
            raise ValueError("Scene metadata has no court_reference mapping.")
        camera_ids = reference.get("camera_ids")
        fits = reference.get("camera_fits")
        if not isinstance(camera_ids, list) or not isinstance(fits, list):
            raise ValueError(
                "Scene metadata court_reference must list camera_ids and camera_fits."
            )
        if len(camera_ids) != len(fits):
            raise ValueError(
                "court_reference camera_ids and camera_fits differ in length."
            )
        if camera_id not in camera_ids:
            raise ValueError(
                f"Camera {camera_id!r} is not in court_reference cameras {camera_ids}."
            )
        fit = fits[camera_ids.index(camera_id)]
        return cls(
            K=np.asarray(fit["K"], dtype=np.float64),
            R=np.asarray(fit["R"], dtype=np.float64),
            t=np.asarray(fit["t"], dtype=np.float64).reshape(3),
            width=width,
            height=height,
        )


@dataclass(frozen=True, slots=True)
class BlenderCameraPose:
    """Blender camera object/data settings reproducing a :class:`PinholeCamera`.

    ``rotation_matrix`` is the camera-to-world rotation of the Blender camera
    object (local -Z forward, +Y up). The lens uses a horizontal sensor fit.
    """

    location: NDArray[np.float64]
    rotation_matrix: NDArray[np.float64]
    lens_mm: float
    sensor_width_mm: float
    resolution_x: int
    resolution_y: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "location": self.location.tolist(),
            "rotation_matrix": self.rotation_matrix.tolist(),
            "lens_mm": self.lens_mm,
            "sensor_width_mm": self.sensor_width_mm,
            "sensor_fit": "HORIZONTAL",
            "resolution_x": self.resolution_x,
            "resolution_y": self.resolution_y,
        }


def to_blender_camera(
    camera: PinholeCamera, *, sensor_width_mm: float
) -> BlenderCameraPose:
    """Convert a centered, square-pixel pinhole camera to a Blender camera pose.

    Blender expresses an off-center principal point through lens shift and
    non-square pixels through pixel aspect; neither is needed by the pipeline
    fits (``fit_camera`` pins the principal point to the image center with
    ``fx == fy``), so both are rejected instead of approximated.
    """
    if not np.isfinite(sensor_width_mm) or sensor_width_mm <= 0.0:
        raise ValueError("sensor_width_mm must be finite and positive.")
    K = camera.K
    if max(abs(K[0, 1]), abs(K[1, 0])) > 1e-9 or not np.isclose(
        K[0, 0], K[1, 1], rtol=1e-9
    ):
        raise ValueError(
            "Blender export requires zero skew and square pixels (fx == fy)."
        )
    if not (
        np.isclose(K[0, 2], camera.width / 2.0)
        and np.isclose(K[1, 2], camera.height / 2.0)
    ):
        raise ValueError(
            "Blender export requires the principal point at the image center; "
            f"got ({K[0, 2]}, {K[1, 2]}) for {camera.width}x{camera.height}."
        )
    return BlenderCameraPose(
        location=camera.center,
        rotation_matrix=camera.R.T @ _OPENCV_TO_BLENDER_CAMERA,
        lens_mm=float(K[0, 0] * sensor_width_mm / camera.width),
        sensor_width_mm=sensor_width_mm,
        resolution_x=camera.width,
        resolution_y=camera.height,
    )


__all__ = ["BlenderCameraPose", "PinholeCamera", "to_blender_camera"]
