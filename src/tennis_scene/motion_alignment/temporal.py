"""CPU COCO17 placement: fixed articulation/scale, temporal position and yaw."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, csr_matrix, vstack

FloatArray: TypeAlias = NDArray[np.float64]
BODY_EDGES = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


class PlacementRejection(IntEnum):
    INSUFFICIENT_JOINTS = 101
    UNOBSERVABLE_YAW = 102
    INSUFFICIENT_SCALE = 103
    SCALE_OUT_OF_RANGE = 104
    SOLVER_FAILED = 105


@dataclass(frozen=True)
class TemporalPlacementConfig:
    data_sigma_m: float = 0.1
    root_acceleration_sigma: float = 10.0
    yaw_acceleration_sigma: float = 20.0
    temporal_weight: float = 0.05
    min_joints: int = 5
    min_scale_pairs: int = 6
    min_scale: float = 0.7
    max_scale: float = 1.4
    max_reprojection_rms_px: float = 10.0
    reprojection_weight_sigma_px: float = 5.0
    max_nfev: int = 200

    def __post_init__(self) -> None:
        for name in (
            "data_sigma_m",
            "root_acceleration_sigma",
            "yaw_acceleration_sigma",
            "temporal_weight",
            "min_scale",
            "max_scale",
            "max_reprojection_rms_px",
            "reprojection_weight_sigma_px",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"placement.{name} must be finite and positive")
        for name in ("min_joints", "min_scale_pairs", "max_nfev"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"placement.{name} must be a positive integer")
        if (
            not 3 <= self.min_joints <= 17
            or self.min_scale_pairs < 3
            or self.min_scale >= self.max_scale
        ):
            raise ValueError("Invalid placement support/scale bounds")


class PlacementUnavailable(ValueError):
    def __init__(self, reason: PlacementRejection, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class TemporalPlacement:
    hip_position: FloatArray
    yaw_correction: FloatArray
    joints: FloatArray
    objective: float
    optimality: float
    nfev: int


def _validate(source: FloatArray, target: FloatArray, weights: FloatArray) -> None:
    if (
        source.ndim != 3
        or source.shape[1:] != (17, 3)
        or target.shape != source.shape
        or weights.shape != source.shape[:2]
    ):
        raise ValueError(
            "COCO17 placement expects matching (T,17,3) points and (T,17) weights"
        )
    if (
        not np.isfinite(source).all()
        or not np.isfinite(weights).all()
        or (weights < 0).any()
    ):
        raise ValueError("Source/weights must be finite and weights nonnegative")
    if not np.isfinite(target[weights > 0]).all():
        raise ValueError("Active target joints must be finite")


def initial_placement(
    source: FloatArray,
    target: FloatArray,
    weights: FloatArray,
    config: TemporalPlacementConfig,
) -> tuple[FloatArray, FloatArray, NDArray[np.uint8]]:
    """Weighted horizontal Procrustes; never interpolate invalid frame estimates."""
    _validate(source, target, weights)
    count = len(source)
    yaw = np.zeros(count)
    root = np.zeros((count, 3))
    reasons: NDArray[np.uint8] = np.full(
        count, int(PlacementRejection.INSUFFICIENT_JOINTS), np.uint8
    )
    for t in np.flatnonzero((weights > 0).sum(-1) >= config.min_joints):
        good = weights[t] > 0
        x, y, w = source[t, good], target[t, good], weights[t, good]
        mx, my = np.average(x, axis=0, weights=w), np.average(y, axis=0, weights=w)
        dx, dy = x - mx, y - my
        a = float(np.sum(w * (dx[:, 0] * dy[:, 0] + dx[:, 1] * dy[:, 1])))
        b = float(np.sum(w * (dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0])))
        if np.hypot(a, b) / w.sum() < 1e-6:
            reasons[t] = int(PlacementRejection.UNOBSERVABLE_YAW)
            continue
        yaw[t] = np.arctan2(b, a)
        c, s = np.cos(yaw[t]), np.sin(yaw[t])
        root[t] = my - [c * mx[0] - s * mx[1], s * mx[0] + c * mx[1], mx[2]]
        reasons[t] = 0
    return yaw, root, reasons


def estimate_body_scale(
    source: FloatArray,
    target: FloatArray,
    weights: FloatArray,
    config: TemporalPlacementConfig,
) -> tuple[float, int]:
    """One median paired-bone ratio for every accepted segment of one identity."""
    _validate(source, target, weights)
    ratios: list[float] = []
    for a, b in BODY_EDGES:
        active = (weights[:, a] > 0) & (weights[:, b] > 0)
        src = np.linalg.norm(source[:, a] - source[:, b], axis=-1)
        dst = np.linalg.norm(target[:, a] - target[:, b], axis=-1)
        active &= (src > 0.1) & np.isfinite(dst) & (dst > 0.1)
        ratios.extend((dst[active] / src[active]).tolist())
    if len(ratios) < config.min_scale_pairs:
        raise PlacementUnavailable(
            PlacementRejection.INSUFFICIENT_SCALE,
            "Not enough paired bones for a person-level scale",
        )
    scale = float(np.median(ratios))
    if not config.min_scale <= scale <= config.max_scale:
        raise PlacementUnavailable(
            PlacementRejection.SCALE_OUT_OF_RANGE,
            f"Body scale {scale:.6g} is outside configured bounds",
        )
    return scale, len(ratios)


def _smooth_matrix(
    count: int, fps: float, config: TemporalPlacementConfig
) -> csr_matrix:
    if count < 3:
        return csr_matrix((0, count * 4))
    centers = np.arange(count - 2)
    row: list[int] = []
    col: list[int] = []
    value: list[float] = []
    for component in range(4):
        sigma = (
            config.yaw_acceleration_sigma
            if component == 0
            else config.root_acceleration_sigma
        )
        denominator = (count - 2) * (1 if component == 0 else 3)
        factor = math.sqrt(2 * config.temporal_weight / denominator) * fps**2 / sigma
        for shift, sign in ((0, 1.0), (1, -2.0), (2, 1.0)):
            row.extend((4 * centers + component).tolist())
            col.extend((4 * (centers + shift) + component).tolist())
            value.extend([factor * sign] * len(centers))
    return csr_matrix((value, (row, col)), shape=(4 * (count - 2), 4 * count))


class _Objective:
    def __init__(
        self,
        source: FloatArray,
        target: FloatArray,
        weights: FloatArray,
        fps: float,
        config: TemporalPlacementConfig,
    ) -> None:
        self.count = len(source)
        self.frame, joint = np.nonzero(weights > 0)
        self.source = source[self.frame, joint]
        self.target = target[self.frame, joint]
        self.weight = np.sqrt(weights[self.frame, joint] / weights.sum())
        self.sigma = config.data_sigma_m
        self.smooth = _smooth_matrix(self.count, fps, config)
        self.rows = np.broadcast_to(
            np.arange(3 * len(self.frame)).reshape(-1, 3, 1), (len(self.frame), 3, 4)
        ).ravel()
        self.cols = np.broadcast_to(
            4 * self.frame[:, None, None] + np.arange(4), (len(self.frame), 3, 4)
        ).ravel()

    def terms(
        self, parameters: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        p = parameters.reshape(-1, 4)[self.frame]
        c, s = np.cos(p[:, 0]), np.sin(p[:, 0])
        x = self.source
        rotated = np.stack(
            (c * x[:, 0] - s * x[:, 1], s * x[:, 0] + c * x[:, 1], x[:, 2]), -1
        )
        residual = (rotated + p[:, 1:] - self.target) / self.sigma
        norm = np.sqrt(1 + np.sum(residual**2, -1))
        factor = np.sqrt(2 / (norm + 1))
        return residual, norm, factor, rotated

    def residual(self, parameters: FloatArray) -> FloatArray:
        r, _, factor, _ = self.terms(parameters)
        return np.concatenate(
            ((r * (self.weight * factor)[:, None]).ravel(), self.smooth @ parameters)
        )

    def jacobian(self, parameters: FloatArray) -> csr_matrix:
        residual, norm, factor, rotated = self.terms(parameters)
        radial = (
            np.eye(3)[None]
            - np.einsum("ni,nj->nij", residual, residual)
            / (2 * norm * (norm + 1))[:, None, None]
        )
        yaw_derivative = np.stack(
            (-rotated[:, 1], rotated[:, 0], np.zeros(len(rotated))), -1
        )
        derivative = np.concatenate(
            (
                yaw_derivative[..., None],
                np.broadcast_to(np.eye(3), (len(rotated), 3, 3)),
            ),
            -1,
        )
        values = (radial @ derivative) * (self.weight * factor / self.sigma)[
            :, None, None
        ]
        data = coo_matrix(
            (values.ravel(), (self.rows, self.cols)),
            shape=(3 * len(self.frame), 4 * self.count),
        )
        return vstack((data, self.smooth), format="csr")


def fit_temporal_placement(
    source: FloatArray,
    target: FloatArray,
    weights: FloatArray,
    *,
    scale: float,
    fps: float,
    config: TemporalPlacementConfig,
) -> TemporalPlacement:
    """Fit one contiguous supported interval; source is court-oriented, hip-centered.

    The vector pseudo-Huber data loss and acceleration penalties match the
    option-2 experiment. Sparse float64 least squares supplies explicit status.
    Gaps and identity boundaries are the caller's responsibility, not filled here.
    """
    _validate(source, target, weights)
    if (
        not len(source)
        or not math.isfinite(fps)
        or fps <= 0
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError(
            "Nonempty placement and positive finite fps/scale are required"
        )
    scaled = source * scale
    yaw, root, reasons = initial_placement(scaled, target, weights, config)
    if reasons.any():
        raise PlacementUnavailable(
            PlacementRejection(int(reasons[reasons != 0][0])),
            "Unsupported frames must split the placement interval",
        )
    yaw = np.unwrap(yaw)
    objective = _Objective(scaled, target, weights, fps, config)
    initial = np.column_stack((yaw, root)).ravel()
    fit = least_squares(
        objective.residual,
        initial,
        jac=objective.jacobian,
        method="trf",
        tr_solver="lsmr",
        x_scale="jac",
        max_nfev=config.max_nfev,
        ftol=1e-7,
        xtol=1e-7,
        gtol=1e-7,
    )
    if not fit.success or not np.isfinite(fit.x).all():
        raise PlacementUnavailable(
            PlacementRejection.SOLVER_FAILED,
            f"Temporal placement failed: {fit.message}",
        )
    parameters = fit.x.reshape(-1, 4)
    angle, root = parameters[:, 0], parameters[:, 1:]
    c, s = np.cos(angle)[:, None], np.sin(angle)[:, None]
    joints = (
        np.stack(
            (
                c * scaled[..., 0] - s * scaled[..., 1],
                s * scaled[..., 0] + c * scaled[..., 1],
                scaled[..., 2],
            ),
            -1,
        )
        + root[:, None]
    )
    return TemporalPlacement(
        root, angle, joints, float(fit.cost), float(fit.optimality), int(fit.nfev)
    )


@dataclass(frozen=True)
class PlacedTrack:
    hip_position: FloatArray
    yaw_correction: FloatArray
    valid: NDArray[np.bool_]
    reasons: NDArray[np.uint8]
    scale: float | None
    scale_pairs: int
    intervals: list[dict[str, Any]]


def fit_supported_track(
    source: FloatArray,
    target: FloatArray,
    weights: FloatArray,
    segment_ids: NDArray[np.int64],
    *,
    fps: float,
    config: TemporalPlacementConfig,
) -> PlacedTrack:
    """Fit one person at native fps, with one scale and no temporal gap bridges."""
    _validate(source, target, weights)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Placement fps must be finite and positive")
    if segment_ids.shape != (len(source),) or not np.issubdtype(
        segment_ids.dtype, np.integer
    ):
        raise ValueError("Body segment IDs must match the time axis")
    active_weights = np.where(segment_ids[:, None] >= 0, weights, 0)
    _, _, reasons = initial_placement(source, target, active_weights, config)
    good = reasons == 0
    reasons[segment_ids < 0] = 1  # no recovered body; retains the existing v2 code
    position, yaw = np.zeros((len(source), 3)), np.zeros(len(source))
    intervals: list[dict[str, Any]] = []
    try:
        scale, pairs = estimate_body_scale(
            source, target, np.where(good[:, None], active_weights, 0), config
        )
    except PlacementUnavailable as exc:
        reasons[good] = int(exc.reason)
        intervals.append(
            {"status": "rejected", "reason": exc.reason.name, "message": str(exc)}
        )
        return PlacedTrack(
            position, yaw, np.zeros(len(source), bool), reasons, None, 0, intervals
        )
    indices = np.flatnonzero(good)
    boundaries = (
        np.flatnonzero((np.diff(indices) != 1) | (np.diff(segment_ids[indices]) != 0))
        + 1
    )
    for frames in np.split(indices, boundaries):
        if not len(frames):
            continue
        info: dict[str, Any] = {
            "start_frame": int(frames[0]),
            "end_frame": int(frames[-1]) + 1,
            "body_segment": int(segment_ids[frames[0]]),
        }
        try:
            fit = fit_temporal_placement(
                source[frames],
                target[frames],
                active_weights[frames],
                scale=scale,
                fps=fps,
                config=config,
            )
        except PlacementUnavailable as exc:
            reasons[frames] = int(exc.reason)
            info.update(status="rejected", reason=exc.reason.name, message=str(exc))
        else:
            position[frames], yaw[frames] = fit.hip_position, fit.yaw_correction
            info.update(
                status="ok",
                nfev=fit.nfev,
                objective=fit.objective,
                optimality=fit.optimality,
            )
        intervals.append(info)
    return PlacedTrack(position, yaw, reasons == 0, reasons, scale, pairs, intervals)
