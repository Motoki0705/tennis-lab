"""Robust gravity-fixed similarity fit between two motion tracks.

The fit maps a source track (for example a GVHMR world track) onto a target
track (for example PLCS root + heading) with a single similarity transform that
is shared by every frame::

    q = scale * Rz(yaw) @ p + translation
    heading_target = wrap(heading_source + yaw)

Conventions
-----------

* Positions are right-handed, Z-up, metres. Inputs are assumed to already live
  in that frame, so the fit never tilts the motion: it only scales, yaws about
  the gravity axis and translates.
* Headings are Z-up yaw angles in radians and follow the ``Rz`` convention used
  by :func:`src.utils.geometry.matrices.rotation_matrix_z`, i.e.
  ``[x, y, z] -> [c * x - s * y, s * x + c * y, z]``. The rotation is applied on
  the left of the column vector, matching ``q = R @ p``.
* Angles are wrapped to ``[-pi, pi)``.

Objective
---------

The unknowns are ``[log(scale), yaw, translation_x, translation_y,
translation_z]``; ``log(scale)`` is dropped when ``fixed_scale`` is set. The
residual vector contains, in this order:

* three position rows per active observation,
  ``(scale * Rz(yaw) @ p + t - q) / sigma_position``,
* one heading row per active observation,
  ``wrap(h_target - h_source - yaw) / sigma_heading``,
* one prior row, ``sqrt(scale_prior) * log(scale)``, present only when the scale
  is free.

Rows are combined with a Huber loss whose threshold is ``huber_delta`` in units
of normalised residuals. Each row contributes ``weight * rho(residual ** 2)``,
so the weight multiplies the loss value *and* its first two derivatives; the
prior row always uses squared loss. The row weight is the observation weight,
with ``heading_weight`` (``lambda_theta``) folded in for the heading rows:
``lambda_theta`` therefore scales the heading loss once, matching
``lambda_theta * rho(r ** 2)``. Scoring the unscaled residual keeps the Huber
threshold comparable across rows; scaling the residual by ``sqrt(weight)``
before the loss would move the threshold per row and is deliberately not done.

Observations whose weight is zero are dropped from the fit, so missing
observations may hold non-finite values.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias, overload

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

Float64Array: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]
ArrayLikeF64: TypeAlias = NDArray[np.float64] | float

_TWO_PI = 2.0 * math.pi
_TINY = float(np.finfo(np.float64).tiny)
# Weighted variance below this value (square metres) counts as "no spread", so
# the associated scale or yaw is reported as unobservable instead of being fit.
_MIN_VARIANCE = 1e-12
# Least-squares starts strictly inside the scale bounds because SciPy rejects an
# initial point that sits on a bound.
_BOUND_INSET = 1e-9

HEADING_CIRCULAR_MEAN = "heading_circular_mean"
POSITION_SVD_UMEYAMA = "position_svd_umeyama"
POSITION_PROJECTION_RATIO = "position_projection_ratio"
POSITION_SVD_XY_SCALE = "position_svd_xy_scale"
FIXED_SCALE = "fixed_scale"


@overload
def wrap_to_pi(angle: float) -> float: ...


@overload
def wrap_to_pi(angle: Float64Array) -> Float64Array: ...


def wrap_to_pi(angle: ArrayLikeF64) -> float | Float64Array:
    """Wrap angles (scalar or array) into ``[-pi, pi)``."""

    values = np.asarray(angle, dtype=np.float64)
    wrapped: Float64Array = (values + math.pi) % _TWO_PI - math.pi
    if values.ndim == 0:
        return float(wrapped)
    return wrapped


@dataclass(frozen=True, slots=True)
class SimilarityTransform:
    """Gravity-fixed similarity ``q = scale * Rz(yaw) @ p + translation``."""

    scale: float
    yaw: float
    translation: Float64Array

    def __post_init__(self) -> None:
        scale = float(self.scale)
        yaw = float(self.yaw)
        translation = np.asarray(self.translation, dtype=np.float64)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"scale must be finite and positive, got {scale!r}.")
        if not math.isfinite(yaw):
            raise ValueError(f"yaw must be finite, got {yaw!r}.")
        if translation.shape != (3,):
            raise ValueError(
                f"translation must have shape (3,), got {translation.shape}."
            )
        if not np.isfinite(translation).all():
            raise ValueError("translation contains NaN or infinity.")
        frozen = np.ascontiguousarray(translation)
        frozen.setflags(write=False)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "yaw", float(wrap_to_pi(yaw)))
        object.__setattr__(self, "translation", frozen)

    @property
    def log_scale(self) -> float:
        """Natural logarithm of :attr:`scale`."""

        return math.log(self.scale)

    def rotation(self) -> Float64Array:
        """Return the ``(3, 3)`` Z-up rotation matrix of ``yaw``."""

        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        rotation: Float64Array = np.array(
            [
                [cos_yaw, -sin_yaw, 0.0],
                [sin_yaw, cos_yaw, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return rotation

    def apply(self, points: Float64Array) -> Float64Array:
        """Transform points of shape ``(..., 3)``; the last axis is XYZ."""

        values = np.asarray(points, dtype=np.float64)
        if values.ndim == 0 or values.shape[-1:] != (3,):
            raise ValueError(f"points must have shape (..., 3), got {values.shape}.")
        transformed: Float64Array = (
            self.scale * (values @ self.rotation().T) + self.translation
        )
        return transformed

    def apply_heading(self, heading: Float64Array) -> Float64Array:
        """Rotate headings by ``yaw`` and wrap them into ``[-pi, pi)``."""

        return wrap_to_pi(np.asarray(heading, dtype=np.float64) + self.yaw)

    def invert(self) -> SimilarityTransform:
        """Return the exact inverse transform."""

        inverse_scale = 1.0 / self.scale
        rotation = self.rotation()
        translation = -inverse_scale * (rotation.T @ self.translation)
        return SimilarityTransform(
            scale=inverse_scale,
            yaw=-self.yaw,
            translation=translation,
        )

    def as_vector(self) -> Float64Array:
        """Return ``[log(scale), yaw, translation_x, translation_y, translation_z]``."""

        return np.concatenate(
            (
                np.array([self.log_scale, self.yaw], dtype=np.float64),
                self.translation,
            )
        )


@dataclass(frozen=True, slots=True)
class SimilarityConfig:
    """Weights, bounds and robust-loss settings for :func:`fit_similarity`."""

    sigma_position: float = 0.5
    sigma_heading: float = math.pi / 6.0
    heading_weight: float = 1.0
    scale_prior: float = 1.0
    min_scale: float = 0.5
    max_scale: float = 2.0
    fixed_scale: float | None = None
    huber_delta: float = 1.0
    heading_resultant_threshold: float = 0.5
    ftol: float = 1e-10
    xtol: float = 1e-10
    gtol: float = 1e-10
    max_nfev: int = 500

    def __post_init__(self) -> None:
        for name in (
            "sigma_position",
            "sigma_heading",
            "min_scale",
            "max_scale",
            "huber_delta",
            "ftol",
            "xtol",
            "gtol",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}.")
        for name in ("heading_weight", "scale_prior"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be finite and non-negative, got {value!r}."
                )
        if self.min_scale > self.max_scale:
            raise ValueError(
                "min_scale must not exceed max_scale, got "
                f"{self.min_scale!r} > {self.max_scale!r}."
            )
        if self.fixed_scale is not None:
            fixed = float(self.fixed_scale)
            if not math.isfinite(fixed) or fixed <= 0.0:
                raise ValueError(
                    f"fixed_scale must be None or positive, got {fixed!r}."
                )
            if not (self.min_scale <= fixed <= self.max_scale):
                raise ValueError(
                    f"fixed_scale {fixed!r} is outside "
                    f"[{self.min_scale!r}, {self.max_scale!r}]."
                )
        if (
            not math.isfinite(self.heading_resultant_threshold)
            or not 0.0 <= self.heading_resultant_threshold <= 1.0
        ):
            raise ValueError(
                "heading_resultant_threshold must lie in [0, 1], got "
                f"{self.heading_resultant_threshold!r}."
            )
        if self.max_nfev <= 0:
            raise ValueError(f"max_nfev must be positive, got {self.max_nfev!r}.")

    @property
    def free_scale(self) -> bool:
        """Whether the scale is a free parameter."""

        return self.fixed_scale is None


@dataclass(frozen=True, slots=True)
class SimilarityFitDiagnostics:
    """How the fit was initialised and how well the solver finished."""

    initializer: str
    scale_initializer: str
    initial_yaw: float
    initial_scale: float
    initial_scale_clamped: bool
    heading_diff_mean: float | None
    heading_resultant_length: float | None
    position_xy_yaw: float | None
    position_xy_scale: float | None
    n_position_observations: int
    n_heading_observations: int
    free_scale: bool
    fixed_scale: float | None
    jacobian_rank: int
    n_free_parameters: int
    scale_at_lower_bound: bool
    scale_at_upper_bound: bool
    optimality: float
    nfev: int
    success: bool
    status: int
    message: str


@dataclass(frozen=True, slots=True)
class SimilarityFitResult:
    """Fitted transform plus the residuals and diagnostics behind it.

    ``cost`` follows the SciPy convention ``0.5 * sum(row_weight * rho(z))`` over
    the residual vector. ``residuals`` holds the normalised residuals before the
    robust loss is applied, in the order described in the module docstring.
    ``position_mask`` and ``heading_mask`` mark the frames that contributed, so
    callers can expand the packed residual blocks back onto the full track.
    Failures raise instead of returning ``success=False``.
    """

    transform: SimilarityTransform
    diagnostics: SimilarityFitDiagnostics
    residuals: Float64Array
    row_weights: Float64Array
    position_mask: BoolArray
    heading_mask: BoolArray
    cost: float
    success: bool


@dataclass(frozen=True, slots=True)
class _Initialization:
    yaw: float
    scale: float
    translation: Float64Array
    initializer: str
    scale_initializer: str
    clamped: bool
    heading_diff_mean: float | None
    heading_resultant_length: float | None
    position_xy_yaw: float | None
    position_xy_scale: float | None


def _as_positions(value: object, *, name: str) -> Float64Array:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {array.shape}.")
    return array


def _as_headings(value: object, *, name: str) -> Float64Array:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape (N,), got {array.shape}.")
    return array


def _as_weights(value: object, *, name: str, size: int) -> Float64Array:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity.")
    if (array < 0.0).any():
        raise ValueError(f"{name} must be non-negative.")
    return array


def _require_finite_active(
    values: Float64Array,
    *,
    name: str,
    active: BoolArray,
) -> None:
    if active.any() and not np.isfinite(values[active]).all():
        raise ValueError(
            f"{name} contains NaN or infinity at an observation with positive "
            "weight; a missing observation must carry zero weight."
        )


def _weighted_mean(values: Float64Array, weights: Float64Array) -> Float64Array:
    mean: Float64Array = (values * weights[:, None]).sum(axis=0) / weights.sum()
    return mean


def _z_rotation(yaw: float) -> Float64Array:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    rotation: Float64Array = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return rotation


def _heading_circular_mean(
    differences: Float64Array,
    weights: Float64Array,
) -> tuple[float, float]:
    """Return the weighted circular mean and its resultant length."""

    cos_sum = float(np.sum(weights * np.cos(differences)))
    sin_sum = float(np.sum(weights * np.sin(differences)))
    total = float(np.sum(weights))
    mean = float(wrap_to_pi(math.atan2(sin_sum, cos_sum)))
    resultant = math.hypot(cos_sum, sin_sum) / total
    return mean, resultant


def _xy_umeyama(
    source_xy: Float64Array,
    target_xy: Float64Array,
    weights: Float64Array,
) -> tuple[float, float] | None:
    """Weighted 2D Umeyama fit returning ``(yaw, scale)`` for a proper rotation."""

    source_centered = source_xy - _weighted_mean(source_xy, weights)
    target_centered = target_xy - _weighted_mean(target_xy, weights)
    variance = float(np.sum(weights * (source_centered**2).sum(axis=1)))
    if variance <= _MIN_VARIANCE:
        return None
    cross: Float64Array = target_centered.T @ (weights[:, None] * source_centered)
    left, singular_values, right_t = np.linalg.svd(cross)
    determinant = float(np.linalg.det(left @ right_t))
    signed = 1.0 if determinant >= 0.0 else -1.0
    rotation: Float64Array = left @ np.diag(np.array([1.0, signed])) @ right_t
    scale = float((singular_values[0] + signed * singular_values[1]) / variance)
    if not math.isfinite(scale) or scale <= 0.0:
        return None
    yaw = float(wrap_to_pi(math.atan2(rotation[1, 0], rotation[0, 0])))
    return yaw, scale


def _projection_scale(
    source: Float64Array,
    target: Float64Array,
    weights: Float64Array,
    rotation: Float64Array,
) -> float | None:
    """Least-squares scale for a fixed rotation, or ``None`` when unobservable."""

    source_centered = source - _weighted_mean(source, weights)
    target_centered = target - _weighted_mean(target, weights)
    rotated = source_centered @ rotation.T
    denominator = float(np.sum(weights * (rotated**2).sum(axis=1)))
    if denominator <= _MIN_VARIANCE:
        return None
    numerator = float(np.sum(weights * (rotated * target_centered).sum(axis=1)))
    scale = numerator / denominator
    if not math.isfinite(scale) or scale <= 0.0:
        return None
    return scale


def _initialization(
    source: Float64Array,
    target: Float64Array,
    source_heading: Float64Array,
    target_heading: Float64Array,
    position_weights: Float64Array,
    heading_weights: Float64Array,
    config: SimilarityConfig,
) -> _Initialization:
    """Choose the starting yaw, scale and translation with recorded reasons."""

    position_active = position_weights > 0.0
    heading_active = heading_weights > 0.0
    source_active = source[position_active]
    target_active = target[position_active]
    weights_active = position_weights[position_active]

    heading_diff_mean: float | None = None
    heading_resultant: float | None = None
    if heading_active.any():
        differences = wrap_to_pi(
            target_heading[heading_active] - source_heading[heading_active]
        )
        heading_diff_mean, heading_resultant = _heading_circular_mean(
            differences, heading_weights[heading_active]
        )

    position_xy_yaw: float | None = None
    position_xy_scale: float | None = None
    umeyama = _xy_umeyama(source_active[:, :2], target_active[:, :2], weights_active)
    if umeyama is not None:
        position_xy_yaw, position_xy_scale = umeyama

    if (
        heading_diff_mean is not None
        and heading_resultant is not None
        and heading_resultant >= config.heading_resultant_threshold
    ):
        yaw = heading_diff_mean
        initializer = HEADING_CIRCULAR_MEAN
    elif position_xy_yaw is not None:
        yaw = position_xy_yaw
        initializer = POSITION_SVD_UMEYAMA
    else:
        raise ValueError(
            "cannot initialise yaw: heading observations have insufficient "
            "resultant length and the active positions have no horizontal spread."
        )

    rotation = _z_rotation(yaw)
    scale: float | None
    if config.fixed_scale is not None:
        scale = float(config.fixed_scale)
        scale_initializer = FIXED_SCALE
    else:
        scale = _projection_scale(
            source_active, target_active, weights_active, rotation
        )
        scale_initializer = POSITION_PROJECTION_RATIO
        if scale is None and position_xy_scale is not None:
            scale = position_xy_scale
            scale_initializer = POSITION_SVD_XY_SCALE
        if scale is None:
            raise ValueError(
                "cannot initialise a free scale: the active positions carry no "
                "usable spread and no alternative position SVD scale exists."
            )

    clamped = False
    if scale < config.min_scale:
        scale = config.min_scale
        clamped = True
    elif scale > config.max_scale:
        scale = config.max_scale
        clamped = True

    target_mean = _weighted_mean(target_active, weights_active)
    source_mean = _weighted_mean(source_active, weights_active)
    translation = target_mean - scale * (rotation @ source_mean)

    return _Initialization(
        yaw=yaw,
        scale=float(scale),
        translation=translation.astype(np.float64, copy=False),
        initializer=initializer,
        scale_initializer=scale_initializer,
        clamped=clamped,
        heading_diff_mean=heading_diff_mean,
        heading_resultant_length=heading_resultant,
        position_xy_yaw=position_xy_yaw,
        position_xy_scale=position_xy_scale,
    )


def _parameters_from(
    yaw: float,
    scale: float,
    translation: Float64Array,
    *,
    free_scale: bool,
) -> Float64Array:
    if free_scale:
        return np.concatenate(
            (
                np.array([math.log(scale), yaw], dtype=np.float64),
                translation,
            )
        )
    return np.concatenate((np.array([yaw], dtype=np.float64), translation))


def _split_parameters(
    parameters: Float64Array,
    *,
    free_scale: bool,
    config: SimilarityConfig,
) -> tuple[float, float, Float64Array]:
    """Return ``(scale, yaw, translation)`` for the parameter vector."""

    if free_scale:
        return (
            math.exp(float(parameters[0])),
            float(parameters[1]),
            parameters[2:5],
        )
    if config.fixed_scale is None:
        raise ValueError("fixed_scale is required when free_scale is False.")
    return (float(config.fixed_scale), float(parameters[0]), parameters[1:4])


def _residuals(
    parameters: Float64Array,
    source: Float64Array,
    target: Float64Array,
    heading_differences: Float64Array,
    *,
    free_scale: bool,
    config: SimilarityConfig,
) -> Float64Array:
    scale, yaw, translation = _split_parameters(
        parameters, free_scale=free_scale, config=config
    )

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    rotated = np.empty_like(source)
    rotated[:, 0] = cos_yaw * source[:, 0] - sin_yaw * source[:, 1]
    rotated[:, 1] = sin_yaw * source[:, 0] + cos_yaw * source[:, 1]
    rotated[:, 2] = source[:, 2]
    position_residual = (scale * rotated + translation - target) / config.sigma_position
    heading_residual = wrap_to_pi(heading_differences - yaw) / config.sigma_heading
    parts = [position_residual.reshape(-1), heading_residual]
    if free_scale:
        prior = math.sqrt(config.scale_prior) * float(parameters[0])
        parts.append(np.array([prior], dtype=np.float64))
    return np.concatenate(parts)


def _jacobian(
    parameters: Float64Array,
    source: Float64Array,
    heading_differences: Float64Array,
    *,
    free_scale: bool,
    config: SimilarityConfig,
) -> Float64Array:
    scale, yaw, _ = _split_parameters(parameters, free_scale=free_scale, config=config)
    n_positions = source.shape[0]
    n_rows = 3 * n_positions + heading_differences.size + (1 if free_scale else 0)
    n_parameters = 5 if free_scale else 4
    jacobian: Float64Array = np.zeros((n_rows, n_parameters), dtype=np.float64)

    if free_scale:
        yaw_index = 1
        translation_index = 2
        log_scale_index = 0
    else:
        yaw_index = 0
        translation_index = 1
        log_scale_index = -1

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    rotated = np.empty_like(source)
    rotated[:, 0] = cos_yaw * source[:, 0] - sin_yaw * source[:, 1]
    rotated[:, 1] = sin_yaw * source[:, 0] + cos_yaw * source[:, 1]
    rotated[:, 2] = source[:, 2]
    # d/dyaw of Rz(yaw) @ p is Rz(yaw) @ (z x p).
    yaw_derivative = np.empty_like(source)
    yaw_derivative[:, 0] = -cos_yaw * source[:, 1] - sin_yaw * source[:, 0]
    yaw_derivative[:, 1] = cos_yaw * source[:, 0] - sin_yaw * source[:, 1]
    yaw_derivative[:, 2] = 0.0

    position_rows = slice(0, 3 * n_positions)
    if free_scale:
        jacobian[position_rows, log_scale_index] = (
            scale * rotated / config.sigma_position
        ).reshape(-1)
    jacobian[position_rows, yaw_index] = (
        scale * yaw_derivative / config.sigma_position
    ).reshape(-1)
    # Residual rows are raveled per observation, so the translation columns are
    # the three interleaved identity patterns.
    for axis in range(3):
        jacobian[axis : 3 * n_positions : 3, translation_index + axis] = (
            1.0 / config.sigma_position
        )
    heading_rows = slice(3 * n_positions, n_rows - (1 if free_scale else 0))
    jacobian[heading_rows, yaw_index] = -1.0 / config.sigma_heading
    if free_scale:
        jacobian[-1, log_scale_index] = math.sqrt(config.scale_prior)
    return jacobian


def _row_weights(
    position_weights_active: Float64Array,
    heading_weights_active: Float64Array,
    *,
    free_scale: bool,
) -> Float64Array:
    parts = [
        np.repeat(position_weights_active, 3),
        heading_weights_active,
    ]
    if free_scale:
        parts.append(np.array([1.0], dtype=np.float64))
    return np.concatenate(parts)


def _make_loss(
    row_weights: Float64Array,
    *,
    huber_delta: float,
    has_prior_row: bool,
) -> Callable[[Float64Array], Float64Array]:
    """Row-wise weighted Huber loss for SciPy's custom ``loss`` callable.

    SciPy expects ``(3, m)`` output holding the loss, its first derivative and
    its second derivative with respect to ``z = residual ** 2``.
    """

    delta_squared = huber_delta * huber_delta

    def loss(z: Float64Array) -> Float64Array:
        safe_z = np.maximum(z, _TINY)
        sqrt_z = np.sqrt(safe_z)
        inlier = z <= delta_squared
        first_derivative = np.where(inlier, 1.0, huber_delta / sqrt_z)
        output: Float64Array = np.empty((3, z.size), dtype=np.float64)
        output[0] = np.where(inlier, z, 2.0 * huber_delta * sqrt_z - delta_squared)
        output[1] = first_derivative
        output[2] = np.where(inlier, 0.0, -0.5 * first_derivative / safe_z)
        output *= row_weights
        if has_prior_row:
            output[:, -1] = np.array([z[-1], 1.0, 0.0], dtype=np.float64)
        return output

    return loss


def _column_scaled_rank(jacobian: Float64Array) -> int:
    """Rank of a Jacobian after normalising column magnitudes."""

    norms = np.linalg.norm(jacobian, axis=0)
    if not np.all(norms > 0.0):
        return int(np.count_nonzero(norms > 0.0))
    singular_values = np.linalg.svd(jacobian / norms, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] <= 0.0:
        return 0
    tolerance = (
        max(jacobian.shape) * float(np.finfo(np.float64).eps) * singular_values[0]
    )
    return int(np.count_nonzero(singular_values > tolerance))


def fit_similarity(
    source_position: Float64Array,
    target_position: Float64Array,
    source_heading: Float64Array | None = None,
    target_heading: Float64Array | None = None,
    position_weights: Float64Array | None = None,
    heading_weights: Float64Array | None = None,
    config: SimilarityConfig | None = None,
) -> SimilarityFitResult:
    """Fit one gravity-fixed similarity transform shared by every frame.

    ``source_position`` and ``target_position`` have shape ``(N, 3)``.
    ``source_heading`` and ``target_heading`` have shape ``(N,)`` and are
    optional; omit them (or pass ``heading_weights`` of zeros) when the track
    carries no heading. Weights default to one. Observations with zero weight are
    dropped from the fit, so those rows may hold NaN or infinity.

    The scale is free unless ``config.fixed_scale`` is set. Because the data are
    already Z-up, this function never estimates a tilt. Unobservable parameters
    (for example a free scale on stationary positions, or a yaw with neither
    reliable headings nor horizontal spread) raise ``ValueError``. A solver that
    reports failure raises ``RuntimeError``.
    """

    settings = SimilarityConfig() if config is None else config
    source = _as_positions(source_position, name="source_position")
    target = _as_positions(target_position, name="target_position")
    if source.shape != target.shape:
        raise ValueError(
            "source_position and target_position must have the same shape, got "
            f"{source.shape} and {target.shape}."
        )
    n_frames = source.shape[0]
    if n_frames == 0:
        raise ValueError("at least one frame is required.")
    if settings.free_scale and settings.min_scale >= settings.max_scale:
        raise ValueError(
            "a free scale requires min_scale < max_scale; use fixed_scale for a "
            "single admissible scale."
        )

    if (source_heading is None) != (target_heading is None):
        raise ValueError(
            "source_heading and target_heading must either both be given or both "
            "be omitted."
        )
    if source_heading is None or target_heading is None:
        source_heading_array = np.zeros(n_frames, dtype=np.float64)
        target_heading_array = np.zeros(n_frames, dtype=np.float64)
        default_heading_weights = np.zeros(n_frames, dtype=np.float64)
    else:
        source_heading_array = _as_headings(source_heading, name="source_heading")
        target_heading_array = _as_headings(target_heading, name="target_heading")
        if source_heading_array.shape != (n_frames,) or target_heading_array.shape != (
            n_frames,
        ):
            raise ValueError(
                "headings must have shape (N,) matching the positions, got "
                f"{source_heading_array.shape} and {target_heading_array.shape}."
            )
        default_heading_weights = np.ones(n_frames, dtype=np.float64)

    if position_weights is None:
        position_weight_array = np.ones(n_frames, dtype=np.float64)
    else:
        position_weight_array = _as_weights(
            position_weights, name="position_weights", size=n_frames
        )
    if heading_weights is None:
        heading_weight_array = default_heading_weights
    else:
        heading_weight_array = _as_weights(
            heading_weights, name="heading_weights", size=n_frames
        )
    # Fold the global heading weight into the per-frame weights so that a
    # heading row contributes heading_weight * heading_weights[t] * rho(r ** 2).
    # A zero heading_weight therefore deactivates every heading row, including
    # the finiteness requirement on those observations.
    heading_weight_array = heading_weight_array * settings.heading_weight

    position_active = position_weight_array > 0.0
    heading_active = heading_weight_array > 0.0
    if not position_active.any():
        raise ValueError(
            "at least one position observation with positive weight is required "
            "to determine the translation."
        )
    _require_finite_active(source, name="source_position", active=position_active)
    _require_finite_active(target, name="target_position", active=position_active)
    _require_finite_active(
        source_heading_array, name="source_heading", active=heading_active
    )
    _require_finite_active(
        target_heading_array, name="target_heading", active=heading_active
    )

    source_active = source[position_active]
    target_active = target[position_active]
    position_weights_active = position_weight_array[position_active]
    heading_weights_active = heading_weight_array[heading_active]
    heading_differences = wrap_to_pi(
        target_heading_array[heading_active] - source_heading_array[heading_active]
    )

    initialization = _initialization(
        source,
        target,
        source_heading_array,
        target_heading_array,
        position_weight_array,
        heading_weight_array,
        settings,
    )

    n_parameters = 5 if settings.free_scale else 4
    initial_scale_clamped = initialization.clamped
    if settings.free_scale:
        log_lower = math.log(settings.min_scale)
        log_upper = math.log(settings.max_scale)
        inset = min(_BOUND_INSET, 0.25 * (log_upper - log_lower))
        log_scale = math.log(initialization.scale)
        clipped = min(max(log_scale, log_lower + inset), log_upper - inset)
        initial_scale_clamped = initial_scale_clamped or clipped != log_scale
        parameters = np.array([clipped, initialization.yaw], dtype=np.float64)
        parameters = np.concatenate((parameters, initialization.translation))
        lower: Float64Array = np.array(
            [log_lower, -np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float64
        )
        upper: Float64Array = np.array(
            [log_upper, np.inf, np.inf, np.inf, np.inf], dtype=np.float64
        )
    else:
        parameters = _parameters_from(
            initialization.yaw,
            initialization.scale,
            initialization.translation,
            free_scale=False,
        )
        lower = np.array([-np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float64)
        upper = np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float64)

    def residuals(parameters: Float64Array) -> Float64Array:
        return _residuals(
            parameters,
            source_active,
            target_active,
            heading_differences,
            free_scale=settings.free_scale,
            config=settings,
        )

    def jacobian(parameters: Float64Array) -> Float64Array:
        return _jacobian(
            parameters,
            source_active,
            heading_differences,
            free_scale=settings.free_scale,
            config=settings,
        )

    row_weights = _row_weights(
        position_weights_active,
        heading_weights_active,
        free_scale=settings.free_scale,
    )
    loss = _make_loss(
        row_weights,
        huber_delta=settings.huber_delta,
        has_prior_row=settings.free_scale,
    )

    solution = least_squares(
        residuals,
        parameters,
        jac=jacobian,
        bounds=(lower, upper),
        method="trf",
        loss=loss,
        f_scale=1.0,
        x_scale="jac",
        ftol=settings.ftol,
        xtol=settings.xtol,
        gtol=settings.gtol,
        max_nfev=settings.max_nfev,
    )

    if not bool(solution.success):
        raise RuntimeError(
            "similarity fit did not converge: "
            f"status={int(solution.status)} message={solution.message!r} "
            f"cost={float(solution.cost)!r}."
        )

    scale, yaw, translation = _split_parameters(
        np.asarray(solution.x, dtype=np.float64),
        free_scale=settings.free_scale,
        config=settings,
    )
    transform = SimilarityTransform(
        scale=scale,
        yaw=yaw,
        translation=np.array(translation, copy=True),
    )

    final_jacobian = jacobian(np.asarray(solution.x, dtype=np.float64))
    rank = _column_scaled_rank(final_jacobian)
    if rank < n_parameters:
        raise ValueError(
            "similarity fit is not identifiable: the weighted observations leave "
            f"rank {rank} < {n_parameters} free parameters. A free scale needs "
            "spread in the active positions; yaw needs either reliable headings "
            "or horizontal spread."
        )

    diagnostics = SimilarityFitDiagnostics(
        initializer=initialization.initializer,
        scale_initializer=initialization.scale_initializer,
        initial_yaw=initialization.yaw,
        initial_scale=initialization.scale,
        initial_scale_clamped=initial_scale_clamped,
        heading_diff_mean=initialization.heading_diff_mean,
        heading_resultant_length=initialization.heading_resultant_length,
        position_xy_yaw=initialization.position_xy_yaw,
        position_xy_scale=initialization.position_xy_scale,
        n_position_observations=int(np.count_nonzero(position_active)),
        n_heading_observations=int(np.count_nonzero(heading_active)),
        free_scale=settings.free_scale,
        fixed_scale=settings.fixed_scale,
        jacobian_rank=rank,
        n_free_parameters=n_parameters,
        scale_at_lower_bound=scale <= settings.min_scale * (1.0 + 1e-9),
        scale_at_upper_bound=scale >= settings.max_scale * (1.0 - 1e-9),
        optimality=float(solution.optimality),
        nfev=int(solution.nfev),
        success=bool(solution.success),
        status=int(solution.status),
        message=str(solution.message),
    )
    return SimilarityFitResult(
        transform=transform,
        diagnostics=diagnostics,
        residuals=residuals(np.asarray(solution.x, dtype=np.float64)),
        row_weights=row_weights,
        position_mask=position_active,
        heading_mask=heading_active,
        cost=float(solution.cost),
        success=True,
    )
