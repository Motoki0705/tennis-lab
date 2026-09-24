"""Pure physical scene and camera value objects for PLCS residual learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CameraRig:
    """OpenCV cameras with metre world coordinates and pixel intrinsics."""

    K: NDArray[np.float64]
    R: NDArray[np.float64]
    t: NDArray[np.float64]
    image_size: NDArray[np.int64]

    def __post_init__(self) -> None:
        views = len(self.K)
        for name, shape in (
            ("K", (views, 3, 3)),
            ("R", (views, 3, 3)),
            ("t", (views, 3)),
            ("image_size", (views, 2)),
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, np.ndarray)
                or value.shape != shape
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"Invalid camera {name}, expected finite {shape}")
        if (
            views < 2
            or (self.image_size <= 0).any()
            or (self.K[:, [0, 1], [0, 1]] <= 0).any()
        ):
            raise ValueError(
                "At least two cameras with positive dimensions/focal lengths are required"
            )
        if not np.allclose(self.K[:, 2], [0, 0, 1], atol=1e-7):
            raise ValueError("K must use the OpenCV pinhole convention")
        if not np.allclose(self.R @ self.R.transpose(0, 2, 1), np.eye(3), atol=1e-5):
            raise ValueError("Camera rotations must be orthonormal")
        if not np.allclose(np.linalg.det(self.R), 1, atol=1e-5):
            raise ValueError("Camera rotations must be proper rotations")

    @property
    def matrices(self) -> NDArray[np.float64]:
        return np.asarray(
            self.K @ np.concatenate((self.R, self.t[..., None]), axis=-1),
            dtype=np.float64,
        )

    @property
    def centers(self) -> NDArray[np.float64]:
        return np.asarray(-np.einsum("vji,vj->vi", self.R, self.t), dtype=np.float64)

    def subset(self, indices: NDArray[np.int64]) -> CameraRig:
        return CameraRig(
            self.K[indices], self.R[indices], self.t[indices], self.image_size[indices]
        )


@dataclass(frozen=True)
class CleanResidualScene:
    scene_id: str
    world_m: NDArray[np.float32]
    fps: float
    rig: CameraRig
    source_group: str


@dataclass(frozen=True)
class RealResidualScene:
    observations_px: NDArray[np.float64]
    scores: NDArray[np.float64]
    court_px: NDArray[np.float64]
    court_scores: NDArray[np.float64]
    rig: CameraRig
    fps: float
    metadata: dict[str, Any]
