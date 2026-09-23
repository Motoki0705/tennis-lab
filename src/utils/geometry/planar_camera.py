"""Observation-only pinhole calibration from a known horizontal plane."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar


class PlanarCameraFailure(StrEnum):
    INVALID_INPUT = "invalid_input"
    INSUFFICIENT_POINTS = "insufficient_points"
    DEGENERATE_POINTS = "degenerate_points"
    POSE_FIT_FAILED = "pose_fit_failed"
    OPTIMIZATION_FAILED = "optimization_failed"
    INVALID_FOCAL_LENGTH = "invalid_focal_length"
    INVALID_DEPTH = "invalid_depth"
    CAMERA_BELOW_PLANE = "camera_below_plane"


class PlanarCameraFitError(ValueError):
    """A calibration failure with a stable reason for sampling diagnostics."""

    def __init__(self, reason: PlanarCameraFailure, message: str) -> None:
        self.reason = reason
        super().__init__(message)


@dataclass(frozen=True)
class PlanarCameraFitDiagnostics:
    input_count: int
    used_count: int
    focal_px: float
    focal_bounds_px: tuple[float, float]
    min_depth_m: float
    camera_height_above_plane_m: float
    objective_evaluations: int


@dataclass(frozen=True)
class PlanarCameraFit:
    """One OpenCV camera, ``X_camera = R @ X_world + t``.

    ``rmse_px`` is the root mean square over both pixel coordinate axes,
    matching the existing real-clip court calibration objective.
    """

    K: NDArray[np.float64]
    R: NDArray[np.float64]
    t: NDArray[np.float64]
    rmse_px: float
    used_mask: NDArray[np.bool_]
    diagnostics: PlanarCameraFitDiagnostics

    @property
    def center(self) -> NDArray[np.float64]:
        return np.asarray(-self.R.T @ self.t, dtype=np.float64)


def _require_two_dimensional_support(points: NDArray[np.float64], name: str) -> None:
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    if singular[0] <= 0 or singular[1] <= singular[0] * 1e-4:
        raise PlanarCameraFitError(
            PlanarCameraFailure.DEGENERATE_POINTS,
            f"Camera calibration requires non-collinear {name} coverage",
        )


def fit_planar_camera(
    world_points_m: np.ndarray,
    pixels: np.ndarray,
    scores: np.ndarray,
    image_size: tuple[int, int],
    *,
    min_score: float = 0.3,
    min_points: int = 6,
) -> PlanarCameraFit:
    """Fit a camera using only labelled plane points and pixel observations.

    The known world plane is horizontal (constant Z, metres). Observations
    with zero/low confidence or outside the image are excluded; missing
    pixels may be NaN only when their score is zero. At least six points with
    two-dimensional world and image coverage are required. Camera-side or
    keypoint-order conventions belong to callers, not this geometric core.

    The model matches real court calibration: fx=fy, image-centred principal
    point, zero skew/distortion, bounded scalar focal minimization with
    iterative solvePnP. Confidence selects the subset but does not alter the
    objective's uniform pixel weighting. No camera or object ground truth,
    calibration prior, outlier repair, or alternative solver is consumed.
    """
    xyz = np.asarray(world_points_m, dtype=np.float64)
    observed = np.asarray(pixels, dtype=np.float64)
    confidence = np.asarray(scores, dtype=np.float64)
    size = np.asarray(image_size)
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or observed.shape != (len(xyz), 2)
        or confidence.shape != (len(xyz),)
        or size.shape != (2,)
        or size.dtype.kind not in "iu"
        or (size <= 0).any()
        or not np.isfinite(xyz).all()
        or not np.isfinite(confidence).all()
        or ((confidence < 0) | (confidence > 1)).any()
        or not np.isfinite(min_score)
        or not 0 <= min_score <= 1
        or isinstance(min_points, bool)
        or not isinstance(min_points, int)
        or min_points < 6
    ):
        raise PlanarCameraFitError(
            PlanarCameraFailure.INVALID_INPUT,
            "Expected finite plane points (N,3), pixels (N,2), scores in [0,1], "
            "positive integer image dimensions, min_score in [0,1], min_points>=6",
        )
    if len(xyz) and not np.allclose(xyz[:, 2], xyz[0, 2], rtol=0, atol=1e-8):
        raise PlanarCameraFitError(
            PlanarCameraFailure.INVALID_INPUT,
            "Camera calibration world points must share one horizontal plane",
        )
    finite = np.isfinite(observed).all(axis=1)
    if np.any((confidence > 0) & ~finite):
        raise PlanarCameraFitError(
            PlanarCameraFailure.INVALID_INPUT,
            "Positive-confidence camera calibration observations must be finite",
        )
    used = (
        finite
        & (confidence > 0)
        & (confidence >= min_score)
        & (observed >= 0).all(axis=1)
        & (observed <= size).all(axis=1)
    )
    used_count = int(used.sum())
    if used_count < min_points:
        raise PlanarCameraFitError(
            PlanarCameraFailure.INSUFFICIENT_POINTS,
            f"Camera calibration has {used_count} usable points; requires {min_points}",
        )
    selected_xyz = np.ascontiguousarray(xyz[used])
    selected_pixels = np.ascontiguousarray(observed[used])
    _require_two_dimensional_support(selected_xyz[:, :2], "world-plane")
    _require_two_dimensional_support(selected_pixels, "image")
    width, height = (int(value) for value in size)
    bounds = (width * 0.2, width * 3.0)

    def solve(
        focal: float,
    ) -> tuple[float, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        intrinsic = np.array(
            [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]],
            dtype=np.float64,
        )
        try:
            ok, rotation_vector, translation = cv2.solvePnP(
                selected_xyz,
                selected_pixels,
                intrinsic,
                None,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                raise PlanarCameraFitError(
                    PlanarCameraFailure.POSE_FIT_FAILED,
                    "Planar camera pose fit failed",
                )
            projected, _ = cv2.projectPoints(
                selected_xyz, rotation_vector, translation, intrinsic, None
            )
        except cv2.error as error:
            raise PlanarCameraFitError(
                PlanarCameraFailure.POSE_FIT_FAILED,
                f"Planar camera pose fit failed: {error}",
            ) from error
        mse = float(np.square(projected[:, 0] - selected_pixels).mean())
        if (
            not np.isfinite(mse)
            or not np.isfinite(rotation_vector).all()
            or not np.isfinite(translation).all()
        ):
            raise PlanarCameraFitError(
                PlanarCameraFailure.POSE_FIT_FAILED,
                "Planar camera pose fit produced nonfinite parameters",
            )
        return (
            mse,
            np.asarray(rotation_vector, dtype=np.float64),
            np.asarray(translation, dtype=np.float64),
            intrinsic,
        )

    fit = minimize_scalar(
        lambda focal: solve(float(focal))[0], bounds=bounds, method="bounded"
    )
    if not fit.success or not np.isfinite(fit.x):
        raise PlanarCameraFitError(
            PlanarCameraFailure.OPTIMIZATION_FAILED,
            "Planar camera focal optimization failed",
        )
    focal = float(fit.x)
    boundary_margin = 1e-5 * (bounds[1] - bounds[0])
    if not bounds[0] + boundary_margin < focal < bounds[1] - boundary_margin:
        raise PlanarCameraFitError(
            PlanarCameraFailure.INVALID_FOCAL_LENGTH,
            f"Planar camera focal length {focal} reached allowed bounds {bounds}",
        )
    mse, rotation_vector, translation, intrinsic = solve(focal)
    rotation = np.asarray(cv2.Rodrigues(rotation_vector)[0], dtype=np.float64)
    offset = np.asarray(translation.ravel(), dtype=np.float64)
    center = -rotation.T @ offset
    depth = selected_xyz @ rotation[2] + offset[2]
    if not np.isfinite(depth).all() or (depth <= 1e-6).any():
        raise PlanarCameraFitError(
            PlanarCameraFailure.INVALID_DEPTH,
            "Planar camera fit puts observed plane points behind the camera",
        )
    camera_height = float(center[2] - xyz[0, 2])
    if not np.isfinite(center).all() or camera_height <= 0:
        raise PlanarCameraFitError(
            PlanarCameraFailure.CAMERA_BELOW_PLANE,
            "Planar camera fit places the camera at or below the known plane",
        )
    return PlanarCameraFit(
        K=intrinsic,
        R=rotation,
        t=offset,
        rmse_px=float(np.sqrt(mse)),
        used_mask=used,
        diagnostics=PlanarCameraFitDiagnostics(
            input_count=len(xyz),
            used_count=used_count,
            focal_px=focal,
            focal_bounds_px=bounds,
            min_depth_m=float(depth.min()),
            camera_height_above_plane_m=camera_height,
            objective_evaluations=int(fit.nfev),
        ),
    )
