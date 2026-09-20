"""Deterministic camera candidates and observation-only Court14 calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.tasks.base.triangulation_residual.contracts import CameraRig
from src.utils.geometry.planar_camera import (
    PlanarCameraFit,
    PlanarCameraFitError,
    fit_planar_camera,
)
from src.utils.projection.camera_projector import make_look_at_camera
from src.utils.schema.court import (
    FENCE_HEIGHT,
    NUM_GROUND_COURT_KP,
    STANDARD_COURT_CONFIG,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    court_keypoints_3d,
)


def fixed_six_camera_rig(image_size: tuple[int, int]) -> CameraRig:
    """Four fence corners then two baseline centres, in fixed-layout order.

    The corners are at fence height; the two baseline cameras are at 5 m.
    All cameras look at the court centre with a 60-degree horizontal FOV.
    This factory does not sample random state or modify source scene cameras.
    """
    size = np.asarray(image_size)
    if size.shape != (2,) or size.dtype.kind not in "iu" or (size <= 0).any():
        raise ValueError("Fixed camera image_size must be positive integer (W,H)")
    centres = (
        (X_MIN, Y_MAX, FENCE_HEIGHT),
        (X_MAX, Y_MAX, FENCE_HEIGHT),
        (X_MAX, Y_MIN, FENCE_HEIGHT),
        (X_MIN, Y_MIN, FENCE_HEIGHT),
        (0.0, Y_MAX, 5.0),
        (0.0, Y_MIN, 5.0),
    )
    cameras = [
        make_look_at_camera(
            centre, look_at=(0.0, 0.0, 0.0), image_size=image_size, hfov_deg=60.0
        )
        for centre in centres
    ]
    intrinsics = np.asarray(
        [
            [[camera.f, 0, camera.cx], [0, camera.f, camera.cy], [0, 0, 1]]
            for camera in cameras
        ],
        dtype=np.float64,
    )
    rotations = np.asarray([camera.R.numpy() for camera in cameras], dtype=np.float64)
    positions = np.asarray([camera.C.numpy() for camera in cameras], dtype=np.float64)
    return CameraRig(
        intrinsics,
        rotations,
        -np.einsum("vij,vj->vi", rotations, positions),
        np.broadcast_to(size, (len(cameras), 2)).astype(np.int64).copy(),
    )


@dataclass(frozen=True)
class CourtRigFit:
    """One calibration outcome per input camera; no substituted cameras."""

    fits: tuple[PlanarCameraFit | None, ...]
    failures: tuple[PlanarCameraFitError | None, ...]
    image_size: NDArray[np.int64]

    @property
    def valid_indices(self) -> NDArray[np.int64]:
        return np.flatnonzero([fit is not None for fit in self.fits]).astype(np.int64)

    def subset(self, indices: NDArray[np.int64]) -> CameraRig:
        """Build an explicitly selected, successfully calibrated camera rig."""
        selected = np.asarray(indices)
        if (
            selected.ndim != 1
            or selected.dtype.kind not in "iu"
            or len(selected) < 2
            or len(np.unique(selected)) != len(selected)
            or ((selected < 0) | (selected >= len(self.fits))).any()
        ):
            raise ValueError("Select at least two distinct valid camera indices")
        fits: list[PlanarCameraFit] = []
        for index in selected:
            fit = self.fits[int(index)]
            if fit is None:
                raise ValueError(f"Selected camera {index} has no successful court fit")
            fits.append(fit)
        return CameraRig(
            np.stack([fit.K for fit in fits]),
            np.stack([fit.R for fit in fits]),
            np.stack([fit.t for fit in fits]),
            self.image_size[selected].copy(),
        )


def fit_court_rig(
    court_px: np.ndarray,
    court_scores: np.ndarray,
    image_size: tuple[int, int] | NDArray[np.int64],
    *,
    min_score: float = 0.3,
    min_points: int = 6,
) -> CourtRigFit:
    """Fit each physical-order Court14 observation once, preserving failures.

    ``image_size`` is either shared (W,H) or one (W,H) pair per camera.
    Callers decide which successful cameras to select or whether to redraw;
    this function never substitutes a true camera or retries observations.
    """
    observations = np.asarray(court_px, dtype=np.float64)
    scores = np.asarray(court_scores, dtype=np.float64)
    if (
        observations.ndim != 3
        or observations.shape[1:] != (NUM_GROUND_COURT_KP, 2)
        or scores.shape != observations.shape[:-1]
        or len(observations) == 0
    ):
        raise ValueError(
            "Court rig calibration requires (V,14,2) pixels and (V,14) scores"
        )
    views = len(observations)
    sizes = np.asarray(image_size)
    if sizes.shape == (2,):
        sizes = np.broadcast_to(sizes, (views, 2))
    if sizes.shape != (views, 2) or sizes.dtype.kind not in "iu" or (sizes <= 0).any():
        raise ValueError(
            "Court rig image_size must contain positive integer (W,H) pairs"
        )
    world = (
        court_keypoints_3d(STANDARD_COURT_CONFIG)
        .numpy()[:NUM_GROUND_COURT_KP]
        .astype(np.float64)
    )
    fits: list[PlanarCameraFit | None] = []
    failures: list[PlanarCameraFitError | None] = []
    for pixels, confidence, size in zip(observations, scores, sizes, strict=True):
        try:
            fit = fit_planar_camera(
                world,
                pixels,
                confidence,
                (int(size[0]), int(size[1])),
                min_score=min_score,
                min_points=min_points,
            )
        except PlanarCameraFitError as error:
            fits.append(None)
            failures.append(error)
        else:
            fits.append(fit)
            failures.append(None)
    return CourtRigFit(tuple(fits), tuple(failures), sizes.astype(np.int64).copy())
