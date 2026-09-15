"""Camera parsing and frustum geometry for the shared scene review UI.

Serialized camera parameters use the repository's OpenCV pinhole convention:
``R`` maps world to camera (its rows are the camera axes) and ``C`` is the
camera centre in physical court metres. Frustum corners come from the shared
:mod:`src.utils.rendering.camera_geometry` helper, so the review UI matches the
publication renderer exactly.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from src.utils.projection.camera_projector import Camera
from src.utils.rendering.camera_geometry import camera_frustum_corners

Matrix3 = tuple[tuple[float, float, float], ...]
FrustumEdge = tuple[int, int]

# Centre-to-corner rays first, then the image-plane perimeter, matching the
# segment order documented by ``camera_geometry.camera_frustum_segments``.
FRUSTUM_EDGES: tuple[FrustumEdge, ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 1),
)

_ROTATION_ATOL = 1.0e-5
_REQUIRED_KEYS = ("C", "R", "f", "cx", "cy", "w", "h")


@dataclass(frozen=True, slots=True)
class CameraRecord:
    """One validated scene camera in the physical court frame."""

    index: int
    id: str
    center: tuple[float, float, float]
    rotation: Matrix3
    f: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def image_size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def intrinsics(self) -> NDArray[np.float64]:
        return np.asarray(
            ((self.f, 0.0, self.cx), (0.0, self.f, self.cy), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )

    def camera_to_world(self) -> NDArray[np.float64]:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation.T
        transform[:3, 3] = np.asarray(self.center, dtype=np.float64)
        return transform

    def to_camera(self) -> Camera:
        """Return the shared :class:`Camera` used by the projector utilities."""
        return Camera(
            C=torch.as_tensor(self.center, dtype=torch.float32),
            R=torch.as_tensor(self.rotation, dtype=torch.float32),
            f=self.f,
            cx=self.cx,
            cy=self.cy,
            w=self.width,
            h=self.height,
        )

    def frustum_vertices(self, *, depth: float) -> NDArray[np.float64]:
        return camera_frustum_corners(
            self.intrinsics(),
            self.image_size,
            self.camera_to_world(),
            depth=depth,
        )

    def to_dict(self, *, depth: float) -> dict[str, Any]:
        return {
            "index": self.index,
            "id": self.id,
            "center": list(self.center),
            "rotation": [list(row) for row in self.rotation],
            "intrinsics": self.intrinsics().tolist(),
            "frustum": self.frustum_vertices(depth=depth).tolist(),
            "image_size": [self.width, self.height],
        }


def _finite(value: object, *, label: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number.") from error
    if not np.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    return number


def _vector3(value: object, *, label: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{label} must have shape (3,), got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values.")
    return (float(array[0]), float(array[1]), float(array[2]))


def _matrix3(value: object, *, label: str) -> Matrix3:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 3) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite 3x3 matrix.")
    gram = array @ array.T
    if not np.allclose(gram, np.eye(3), atol=_ROTATION_ATOL, rtol=0.0):
        raise ValueError(f"{label} must be an orthonormal rotation.")
    if not np.isclose(np.linalg.det(array), 1.0, atol=_ROTATION_ATOL, rtol=0.0):
        raise ValueError(f"{label} must be a proper rotation (det +1).")
    return (
        (float(array[0, 0]), float(array[0, 1]), float(array[0, 2])),
        (float(array[1, 0]), float(array[1, 1]), float(array[1, 2])),
        (float(array[2, 0]), float(array[2, 1]), float(array[2, 2])),
    )


def _parse_camera_params(params: object, *, index: int) -> CameraRecord:
    if isinstance(params, str):
        params = json.loads(params)
    if not isinstance(params, Mapping):
        raise ValueError(f"cam_{index}_params must be a JSON object.")
    missing = [key for key in _REQUIRED_KEYS if key not in params]
    if missing:
        raise ValueError(f"cam_{index}_params is missing keys {missing!r}.")
    width = params["w"]
    height = params["h"]
    if isinstance(width, bool) or isinstance(height, bool):
        raise ValueError(f"cam_{index}_params image size must be integers.")
    width_int = int(width)
    height_int = int(height)
    if width_int <= 0 or height_int <= 0:
        raise ValueError(f"cam_{index}_params image size must be positive.")
    return CameraRecord(
        index=index,
        id=f"cam_{index}",
        center=_vector3(params["C"], label=f"cam_{index}_params.C"),
        rotation=_matrix3(params["R"], label=f"cam_{index}_params.R"),
        f=_finite(params["f"], label=f"cam_{index}_params.f"),
        cx=_finite(params["cx"], label=f"cam_{index}_params.cx"),
        cy=_finite(params["cy"], label=f"cam_{index}_params.cy"),
        width=width_int,
        height=height_int,
    )


def parse_cameras(scalars: Mapping[str, Any]) -> tuple[CameraRecord, ...]:
    """Parse and validate every ``cam_<i>_params`` slot in ``scalars``."""
    num_cameras = scalars.get("num_cameras")
    if isinstance(num_cameras, bool) or not isinstance(num_cameras, int):
        raise ValueError("scalars.num_cameras must be an integer.")
    if num_cameras <= 0:
        raise ValueError("scalars.num_cameras must be positive.")
    cameras: list[CameraRecord] = []
    for index in range(num_cameras):
        key = f"cam_{index}_params"
        if key not in scalars:
            raise ValueError(f"scalars is missing required {key!r}.")
        cameras.append(_parse_camera_params(scalars[key], index=index))
    return tuple(cameras)


__all__ = [
    "FRUSTUM_EDGES",
    "CameraRecord",
    "FrustumEdge",
    "Matrix3",
    "parse_cameras",
]
