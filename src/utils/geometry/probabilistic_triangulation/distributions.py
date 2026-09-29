"""CPU distributions in explicit source pixels and world metres."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

FloatArray: TypeAlias = NDArray[np.float64]


def validate_covariance(value: FloatArray, shape: tuple[int, ...]) -> None:
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError("Invalid covariance shape or nonfinite values")
    if not np.allclose(value, value.swapaxes(-1, -2), atol=1e-10, rtol=1e-7):
        raise ValueError("Covariance must be symmetric")
    try:
        np.linalg.cholesky(value)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Covariance must be positive definite") from exc


@dataclass(frozen=True)
class GaussianPrior3D:
    mean: FloatArray
    covariance: FloatArray

    def __post_init__(self) -> None:
        if self.mean.shape != (3,) or not np.isfinite(self.mean).all():
            raise ValueError("Prior mean must be finite (3,)")
        validate_covariance(self.covariance, (3, 3))


@dataclass(frozen=True)
class CameraGMM:
    """One synchronized frame, V cameras, K alternatives; not a detector API."""

    means_px: FloatArray  # V,K,2
    covariance_px2: FloatArray  # V,K,2,2
    weights: FloatArray  # V,K, conditional on presence
    presence: FloatArray  # V, amodal in-image presence (not visibility)

    def __post_init__(self) -> None:
        if self.means_px.ndim != 3 or self.means_px.shape[-1] != 2:
            raise ValueError("Means must have shape (V,K,2)")
        v, k, _ = self.means_px.shape
        if min(v, k) < 1 or not np.isfinite(self.means_px).all():
            raise ValueError("Means must be finite with nonempty V,K")
        validate_covariance(self.covariance_px2, (v, k, 2, 2))
        if self.weights.shape != (v, k) or not np.isfinite(self.weights).all():
            raise ValueError("Invalid mixture weights")
        if (self.weights < 0).any() or not np.allclose(
            self.weights.sum(-1), 1, atol=1e-6
        ):
            raise ValueError("Mixture weights must be nonnegative and sum to one")
        if self.presence.shape != (v,) or not np.isfinite(self.presence).all():
            raise ValueError("Invalid presence shape or nonfinite values")
        if ((self.presence < 0) | (self.presence > 1)).any():
            raise ValueError("Presence must be in [0,1]")
        # Exported float32 softmax rows may miss one by roundoff. Normalize only
        # after the contract check; do not threshold or prune any component.
        object.__setattr__(
            self, "weights", self.weights / self.weights.sum(-1, keepdims=True)
        )


@dataclass(frozen=True)
class GaussianMixture3D:
    means: FloatArray  # M,3
    covariance: FloatArray  # M,3,3
    weights: FloatArray  # M

    def __post_init__(self) -> None:
        if self.means.ndim != 2 or self.means.shape[-1] != 3:
            raise ValueError("3D means must be (M,3)")
        m = len(self.means)
        if m < 1 or not np.isfinite(self.means).all():
            raise ValueError("3D means must be finite and nonempty")
        validate_covariance(self.covariance, (m, 3, 3))
        if self.weights.shape != (m,) or not np.isfinite(self.weights).all():
            raise ValueError("Invalid 3D weights")
        if (self.weights < 0).any() or not np.isclose(self.weights.sum(), 1, atol=1e-6):
            raise ValueError("3D weights must be nonnegative and sum to one")
        object.__setattr__(self, "weights", self.weights / self.weights.sum())

    def log_prob(self, points: FloatArray) -> FloatArray:
        if points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ValueError("Query points must be finite (...,3)")
        delta = points[..., None, :] - self.means
        precision = np.linalg.inv(self.covariance)
        quadratic = np.einsum("...mi,mij,...mj->...m", delta, precision, delta)
        with np.errstate(divide="ignore"):
            component = np.log(self.weights) - 0.5 * (
                quadratic
                + np.linalg.slogdet(self.covariance)[1]
                + 3 * np.log(2 * np.pi)
            )
        return np.asarray(logsumexp(component, axis=-1), dtype=np.float64)

    def sample(self, count: int, rng: np.random.Generator) -> FloatArray:
        if count < 1:
            raise ValueError("Sample count must be positive")
        index = rng.choice(len(self.weights), size=count, p=self.weights)
        normal = rng.standard_normal((count, 3))
        return self.means[index] + np.einsum(
            "nij,nj->ni", np.linalg.cholesky(self.covariance[index]), normal
        )

    def moments(self) -> tuple[FloatArray, FloatArray]:
        mean = self.weights @ self.means
        centered = self.means - mean
        covariance = np.einsum("m,mij->ij", self.weights, self.covariance) + np.einsum(
            "m,mi,mj->ij", self.weights, centered, centered
        )
        return mean, covariance


def camera_subsets(presence: FloatArray) -> list[tuple[NDArray[np.int64], float]]:
    """Positive-mass independent Bernoulli subsets; exact zeros/ones stay exact."""
    choices = [(False, True) if 0 < p < 1 else (bool(p),) for p in presence]
    result = []
    for selected in product(*choices):
        mask = np.asarray(selected, dtype=bool)
        probability = float(np.prod(np.where(mask, presence, 1 - presence)))
        result.append((np.flatnonzero(mask), probability))
    return result

