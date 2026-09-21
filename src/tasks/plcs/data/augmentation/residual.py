"""Six-view draws whose estimated cameras come only from noisy Court14.

Component RNG streams are allocated before sampling any component. Disabling
one corruption does not advance another component's random stream. Candidate
selection/retries belong to the dataset, after all six cameras are fitted once.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from src.tasks.plcs.configuration_contracts import ResidualAugmentationConfig
from src.tasks.plcs.data.augmentation.persistent_pose import (
    PersistentEvent,
    corrupt_persistent,
)
from src.tasks.plcs.data.residual_types import CameraRig
from src.utils.geometry.planar_camera import (
    PlanarCameraFit,
    PlanarCameraFitError,
    fit_planar_camera,
)
from src.utils.geometry.triangulation import project_multiview
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


@dataclass(frozen=True)
class CorruptedObservations:
    observations_px: np.ndarray
    scores: np.ndarray
    court_px: np.ndarray
    court_scores: np.ndarray
    estimated_rig: CameraRig
    true_rig: CameraRig
    clean_uv: np.ndarray
    clean_visible: np.ndarray
    severity: float


def _distort(pixel: np.ndarray, rig: CameraRig, radial: np.ndarray) -> np.ndarray:
    shape = (len(rig.K),) + (1,) * (pixel.ndim - 2) + (2,)
    principal = rig.K[:, :2, 2].reshape(shape)
    focal = rig.K[:, [0, 1], [0, 1]].reshape(shape)
    xy = (pixel - principal) / focal
    k = radial.reshape((len(rig.K),) + (1,) * (pixel.ndim - 1))
    distorted: np.ndarray = np.asarray(
        principal + focal * xy * (1 + k * np.sum(xy**2, axis=-1, keepdims=True))
    )
    return distorted


class ErrorFamily(IntEnum):
    CLEAN = 0
    CALIBRATION = 1
    OBSERVATION = 2
    TEMPORAL = 3
    PERSISTENT = 4
    COMBINED = 5


@dataclass(frozen=True)
class CorruptedCandidates:
    observations_px: NDArray[np.float64]
    scores: NDArray[np.float64]
    court_px: NDArray[np.float64]
    court_scores: NDArray[np.float64]
    true_rig: CameraRig
    clean_uv: NDArray[np.float32]
    clean_visible: NDArray[np.bool_]
    severity: float
    family: int
    court_fit: CourtRigFit
    persistent_mask: NDArray[np.bool_]
    persistent_kind: NDArray[np.uint8]
    persistent_source_missing: NDArray[np.bool_]
    persistent_events: tuple[PersistentEvent, ...]
    radial_coefficients: NDArray[np.float64]
    dropout_mask: NDArray[np.bool_]
    time_shifts: NDArray[np.int64]

    @property
    def valid_indices(self) -> NDArray[np.int64]:
        return self.court_fit.valid_indices

    def subset(self, indices: NDArray[np.int64]) -> CorruptedObservations:
        """Select one fitted camera subset without redrawing any observation."""
        estimated = self.court_fit.subset(indices)
        return CorruptedObservations(
            self.observations_px[indices],
            self.scores[indices],
            self.court_px[indices],
            self.court_scores[indices],
            estimated,
            self.true_rig.subset(indices),
            self.clean_uv[indices],
            self.clean_visible[indices],
            self.severity,
        )


def _sample_family(
    rng: np.random.Generator,
    corruption: ResidualAugmentationConfig,
    config: ResidualAugmentationConfig,
) -> tuple[ErrorFamily, float]:
    if config.error_mode != "mixed":
        try:
            family = ErrorFamily[config.error_mode.upper()]
        except KeyError as error:
            raise ValueError(f"Unknown corruption mode: {config.error_mode}") from error
        return family, 0.0 if family == ErrorFamily.CLEAN else 1.0
    mixture = rng.random()
    if mixture < corruption.clean_probability:
        return ErrorFamily.CLEAN, 0.0
    if mixture < corruption.clean_probability + corruption.hard_probability:
        return ErrorFamily.COMBINED, float(rng.uniform(2, 4))
    return ErrorFamily(int(rng.integers(1, 6))), 1.0


def _true_camera_rig(
    base: CameraRig,
    rng: np.random.Generator,
    corruption: ResidualAugmentationConfig,
    config: ResidualAugmentationConfig,
) -> CameraRig:
    centers = base.centers.copy()
    centers[:, :2] += (
        rng.uniform(-1, 1, (len(centers), 2)) * config.true_camera_position_jitter_m
    )
    centers[:, 2] += (
        rng.uniform(-1, 1, len(centers)) * config.true_camera_height_jitter_m
    )
    if (centers[:, 2] <= 0).any():
        raise ValueError("True-camera jitter placed a camera below the court plane")
    forward = -centers / np.linalg.norm(centers, axis=-1, keepdims=True)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    norm = np.linalg.norm(right, axis=-1, keepdims=True)
    if (norm <= 1e-8).any():
        raise ValueError(
            "True-camera layout cannot look vertically at the court centre"
        )
    right /= norm
    rotation = np.stack((right, np.cross(forward, right), forward), axis=1)
    intrinsics = base.K.copy()
    zoom = np.exp(
        rng.uniform(
            np.log(corruption.focal_scale_min),
            np.log(corruption.focal_scale_max),
            len(centers),
        )
    )
    intrinsics[:, 0, 0] *= zoom
    intrinsics[:, 1, 1] *= zoom
    return CameraRig(
        intrinsics,
        rotation,
        -np.einsum("vij,vj->vi", rotation, centers),
        base.image_size.copy(),
    )


def _inside(points: np.ndarray, image_size: np.ndarray) -> NDArray[np.bool_]:
    sizes = image_size.reshape((len(image_size),) + (1,) * (points.ndim - 2) + (2,))
    return np.asarray(
        np.isfinite(points).all(axis=-1)
        & (points >= 0).all(axis=-1)
        & (points <= sizes).all(axis=-1),
        dtype=bool,
    )


def _court_observations(
    court: np.ndarray,
    depth: np.ndarray,
    rig: CameraRig,
    radial: np.ndarray,
    rng: np.random.Generator,
    config: ResidualAugmentationConfig,
    *,
    severity: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    pixel_scale = rig.image_size[:, 1] / 1080
    obs = _distort(court, rig, radial)
    obs += (
        rng.normal(size=court.shape)
        * config.court_noise_px
        * severity
        * pixel_scale[:, None, None]
    )
    obs += (
        rng.normal(size=(len(court), 1, 2))
        * config.court_bias_px
        * severity
        * pixel_scale[:, None, None]
    )
    outlier = rng.random(court.shape[:-1]) < config.court_outlier_probability * min(
        severity, 2
    )
    obs += (
        outlier[..., None]
        * rng.normal(size=court.shape)
        * config.court_outlier_sigma_px
        * severity
        * pixel_scale[:, None, None]
    )
    scores = np.where(outlier, rng.uniform(0.1, 0.85, court.shape[:-1]), 1.0)
    missing = rng.random(scores.shape) < config.court_dropout_probability * min(
        severity, 2
    )
    scores[missing | (depth <= 0.05) | ~_inside(obs, rig.image_size)] = 0
    obs[scores == 0] = np.nan
    return np.asarray(obs, dtype=np.float64), np.asarray(scores, dtype=np.float64)


def _object_observations(
    reference: np.ndarray,
    clean_visible: np.ndarray,
    image_size: np.ndarray,
    rng: np.random.Generator,
    config: ResidualAugmentationConfig,
    *,
    fps: float,
    severity: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    views, frames, joints = reference.shape[:3]
    obs = reference.copy()
    pixel_scale = image_size[:, 1] / 1080
    joint_scale = np.ones(joints)
    if joints == 17:
        joint_scale[[7, 8, 9, 10, 15, 16]] = 1.6
    white = rng.normal(size=obs.shape) * config.observation_sigma_px * severity
    white *= pixel_scale[:, None, None, None] * joint_scale[None, None, :, None]
    smooth = rng.normal(size=obs.shape)
    rho = np.exp(-1.0 / (fps * 0.2))
    for frame in range(1, frames):
        smooth[:, frame] = (
            rho * smooth[:, frame - 1] + np.sqrt(1 - rho**2) * smooth[:, frame]
        )
    smooth *= config.temporal_sigma_px * severity * pixel_scale[:, None, None, None]
    bias = (
        rng.normal(size=(views, 1, joints, 2))
        * config.view_bias_px
        * severity
        * pixel_scale[:, None, None, None]
    )
    obs += white + smooth + bias
    confidence = np.clip(
        rng.normal(0.85, config.confidence_noise, (views, frames, joints)), 0.35, 1.0
    )
    # Confidence uncertainty exists even without coordinate noise. Otherwise
    # persistent-only examples expose an exact 1.0-versus-<1.0 error label.
    outlier = rng.random(confidence.shape) < config.outlier_probability * min(
        severity, 2
    )
    obs += (
        outlier[..., None]
        * rng.normal(size=obs.shape)
        * config.outlier_sigma_px
        * severity
        * pixel_scale[:, None, None, None]
    )
    confidence = np.where(outlier, rng.uniform(0.1, 0.8, confidence.shape), confidence)
    confidence[~clean_visible | ~_inside(obs, image_size)] = 0
    obs[confidence == 0] = np.nan
    return np.asarray(obs, dtype=np.float64), np.asarray(confidence, dtype=np.float64)


def _missing_mask(
    shape: tuple[int, ...],
    rng: np.random.Generator,
    config: ResidualAugmentationConfig,
    *,
    severity: float,
) -> NDArray[np.bool_]:
    views, frames, joints = shape
    missing = rng.random(shape) < config.dropout_probability * min(severity, 2)
    for view in range(views):
        active = rng.random() < config.burst_probability * min(severity, 2)
        start = int(rng.integers(frames))
        length = int(rng.integers(1, max(1, config.burst_max_frames) + 1))
        whole_view = rng.random() < 0.4
        joint = int(rng.integers(joints))
        if active and config.burst_max_frames > 0:
            if whole_view:
                missing[view, start : start + length] = True
            else:
                missing[view, start : start + length, joint] = True
    return np.asarray(missing, dtype=bool)


def _shift_frames(
    obs: np.ndarray,
    scores: np.ndarray,
    labels: tuple[np.ndarray, ...],
    rng: np.random.Generator,
    config: ResidualAugmentationConfig,
    *,
    severity: float,
) -> NDArray[np.int64]:
    views, frames = obs.shape[:2]
    shifts = np.zeros(views, dtype=np.int64)
    for view in range(views):
        active = rng.random() < config.time_shift_probability * min(severity, 2)
        shift = int(
            rng.integers(
                -config.time_shift_max_frames, config.time_shift_max_frames + 1
            )
        )
        if view == 0 or not active:
            continue
        shifts[view] = shift
        source: NDArray[np.int64] = np.arange(frames, dtype=np.int64) + shift
        inside = (source >= 0) & (source < frames)
        clipped = np.clip(source, 0, frames - 1)
        obs[view] = obs[view, clipped]
        scores[view] = scores[view, clipped] * inside[:, None]
        for label in labels:
            label[view] = label[view, clipped] * inside[:, None]
    return shifts


def corrupt_candidates(
    world_m: np.ndarray,
    base_rig: CameraRig,
    rng: np.random.Generator,
    config: ResidualAugmentationConfig,
    *,
    fps: float,
    fixed_error: tuple[int, float] | None = None,
) -> CorruptedCandidates:
    """Draw one full six-camera observation set and fit its noisy court once.

    Only Court14 pixels, their confidence, and image dimensions cross the
    calibration boundary. True cameras and object GT remain loss/debug targets.
    The caller chooses from ``valid_indices`` using observation geometry only.
    On a retry it passes the first draw's ``fixed_error=(family, severity)`` so
    a failed hard calibration cannot be replaced with a clean/normal family.
    """
    world = np.asarray(world_m, dtype=np.float64)
    if (
        world.ndim != 3
        or world.shape[1:] != (17, 3)
        or len(world) < 1
        or not np.isfinite(world).all()
        or len(base_rig.K) != 6
        or not np.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError(
            "Residual augmentation requires finite COCO17 GT, exactly six base cameras, and positive FPS"
        )
    seeds = rng.integers(0, np.iinfo(np.int64).max, size=8)
    (
        camera_rng,
        mixture_rng,
        radial_rng,
        court_rng,
        object_rng,
        persistent_rng,
        missing_rng,
        shift_rng,
    ) = (np.random.default_rng(int(seed)) for seed in seeds)
    family, severity = _sample_family(mixture_rng, config, config)
    if fixed_error is not None:
        if (
            not isinstance(fixed_error, tuple)
            or len(fixed_error) != 2
            or isinstance(fixed_error[0], bool)
            or not isinstance(fixed_error[0], (int, np.integer))
            or isinstance(fixed_error[1], bool)
            or not isinstance(fixed_error[1], (int, float, np.integer, np.floating))
            or not np.isfinite(fixed_error[1])
        ):
            raise ValueError("fixed_error must be (integer family, finite severity)")
        try:
            family = ErrorFamily(int(fixed_error[0]))
        except ValueError as error:
            raise ValueError("fixed_error family must be in 0..5") from error
        severity = float(fixed_error[1])
        if (family == ErrorFamily.CLEAN and severity != 0) or (
            family != ErrorFamily.CLEAN and severity < 1
        ):
            raise ValueError(
                "fixed_error needs zero severity for clean, at least one otherwise"
            )
    calibration = family in (ErrorFamily.CALIBRATION, ErrorFamily.COMBINED)
    observation = family in (ErrorFamily.OBSERVATION, ErrorFamily.COMBINED)
    temporal = family in (ErrorFamily.TEMPORAL, ErrorFamily.COMBINED)
    persistent = family in (ErrorFamily.PERSISTENT, ErrorFamily.COMBINED)
    true_rig = _true_camera_rig(base_rig, camera_rng, config, config)
    clean, depth = project_multiview(world, true_rig.matrices)
    clean = clean.transpose(2, 0, 1, 3)
    clean_visible = (depth.transpose(2, 0, 1) > 0.05) & _inside(
        clean, true_rig.image_size
    )
    clean_uv = np.where(
        clean_visible[..., None], clean / true_rig.image_size[:, None, None], 0
    ).astype(np.float32)
    radial = radial_rng.normal(
        0, config.radial_std * severity * calibration, len(true_rig.K)
    )
    reference = _distort(clean, true_rig, radial)
    obs, scores = _object_observations(
        reference,
        clean_visible,
        true_rig.image_size,
        object_rng,
        config,
        fps=fps,
        severity=severity * observation,
    )
    court_world = court_keypoints_3d(STANDARD_COURT_CONFIG).numpy().astype(float)[:14]
    court, court_depth = project_multiview(court_world, true_rig.matrices)
    court_px, court_scores = _court_observations(
        court.transpose(1, 0, 2),
        court_depth.T,
        true_rig,
        radial,
        court_rng,
        config,
        severity=severity * calibration,
    )
    # No true camera, world object, clean observation, or previous fit is an
    # argument to this observation-only calibration call.
    court_fit = fit_court_rig(
        court_px,
        court_scores,
        true_rig.image_size,
        min_points=config.calibration_min_points,
    )
    persistent_mask = np.zeros(scores.shape, dtype=bool)
    persistent_kind = np.zeros(scores.shape, dtype=np.uint8)
    source_missing = np.zeros(scores.shape, dtype=bool)
    events: tuple[PersistentEvent, ...] = ()
    if persistent:
        generated = corrupt_persistent(
            obs,
            scores,
            reference,
            clean_visible,
            true_rig.image_size,
            persistent_rng,
            config,
            fps=fps,
            severity=severity,
            force_event=family == ErrorFamily.PERSISTENT,
        )
        obs, scores = generated.observations_px, generated.scores
        persistent_mask, persistent_kind = generated.mask, generated.kind
        source_missing, events = generated.source_missing, generated.events
    missing = _missing_mask(
        scores.shape, missing_rng, config, severity=severity * temporal
    )
    scores[missing] = 0
    shifts = _shift_frames(
        obs,
        scores,
        (persistent_mask, persistent_kind, source_missing, missing),
        shift_rng,
        config,
        severity=severity * temporal,
    )
    invalid = (scores <= 0) | ~_inside(obs, true_rig.image_size)
    scores[invalid] = 0
    obs[invalid] = np.nan
    persistent_mask[invalid] = False
    persistent_kind[invalid] = 0
    return CorruptedCandidates(
        obs,
        scores,
        court_px,
        court_scores,
        true_rig,
        clean_uv,
        clean_visible,
        severity,
        int(family),
        court_fit,
        persistent_mask,
        persistent_kind,
        source_missing,
        events,
        np.asarray(radial, dtype=np.float64),
        missing,
        shifts,
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
