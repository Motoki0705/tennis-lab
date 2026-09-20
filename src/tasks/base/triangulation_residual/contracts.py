"""Explicit physical-court contracts; never the legacy SMPL-root/yaw contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor


@dataclass(frozen=True)
class CameraRig:
    """Fixed OpenCV cameras: X_camera = R @ X_world + t, metres and pixels."""

    K: NDArray[np.float64]  # V,3,3
    R: NDArray[np.float64]  # V,3,3
    t: NDArray[np.float64]  # V,3
    image_size: NDArray[np.int64]  # V,2: width,height

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
    world_m: NDArray[np.float32]  # T,J,3, physical court world; never normalized
    fps: float
    rig: CameraRig
    source_group: str


@dataclass(frozen=True)
class RealResidualScene:
    observations_px: NDArray[np.float64]  # V,T,J,2; missing may be NaN with score=0
    scores: NDArray[np.float64]  # V,T,J
    court_px: NDArray[np.float64]  # V,14,2, PHYSICAL CourtKP14 order
    court_scores: NDArray[np.float64]  # V,14
    rig: CameraRig
    fps: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GeometryInput:
    features: NDArray[np.float32]  # V,T,F, finite model input
    init_world_m: NDArray[np.float32]  # T,J,3, explicit observation-only filled seed
    raw_init_world_m: NDArray[np.float64]  # T,J,3, NaN for untriangulated points
    init_valid: NDArray[np.bool_]  # T,J; true only for raw triangulation
    root_init_m: NDArray[np.float32]  # T,3
    relative_init_m: NDArray[np.float32]  # T,J,3
    time_positions: NDArray[np.float32]  # T, reference ticks = seconds * 30
    view_valid: NDArray[np.bool_]  # V,T; actual view/time, independent of detections
    observations_uv: NDArray[np.float32]  # V,T,J,2
    scores: NDArray[np.float32]  # V,T,J
    reprojected_uv: NDArray[np.float32]  # V,T,J,2
    residual_uv: NDArray[
        np.float32
    ]  # V,T,J,2 = original-reprojected with validity mask
    used_views: NDArray[np.bool_]  # V,T,J
    reprojection_valid: NDArray[np.bool_]  # V,T,J
    ray_angle_deg: NDArray[np.float32]  # T,J; best acute used-view ray angle


def feature_dimension(joints: int) -> int:
    """6 UV + 3 world + 5 confidence/masks + 3 relative + 1 angle per joint;
    court UV/confidence (42), camera (16), root (3) shared per view/time.
    """
    return 18 * joints + 61


def validate_model_inputs(
    features: Tensor,
    view_valid: Tensor,
    time_positions: Tensor,
    *,
    input_dim: int,
) -> None:
    """Validate at the call boundary before the computation-only model forward."""
    if features.ndim != 4 or features.shape[-1] != input_dim:
        raise ValueError("Expected features [B,V,T,F] for this residual profile")
    batch, _, frames, _ = features.shape
    if (
        view_valid.shape != features.shape[:3]
        or view_valid.dtype != torch.bool
        or time_positions.shape != (batch, frames)
    ):
        raise ValueError("Invalid residual view/time contract")
    if (
        features.device != view_valid.device
        or features.device != time_positions.device
    ):
        raise ValueError("Inputs must share a device")
