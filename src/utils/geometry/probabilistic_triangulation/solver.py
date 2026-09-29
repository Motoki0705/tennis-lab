"""Enumerated component products with covariance-weighted nonlinear fitting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.special import logsumexp

from src.utils.geometry.probabilistic_triangulation.distributions import (
    CameraGMM,
    FloatArray,
    GaussianMixture3D,
    GaussianPrior3D,
)
from src.utils.geometry.triangulation import PinholeCamera


@dataclass(frozen=True)
class LaplaceConfig:
    max_components: int
    max_nfev: int

    def __post_init__(self) -> None:
        if self.max_components < 1 or self.max_nfev < 1:
            raise ValueError("Component and optimizer budgets must be positive")


@dataclass(frozen=True)
class ProbabilisticTriangulation:
    distribution: GaussianMixture3D
    camera_subsets: NDArray[np.bool_]  # M,V; same order as mixture components
    prior_only_probability: float


def camera_subsets(presence: FloatArray) -> list[tuple[NDArray[np.int64], float]]:
    """Positive-mass independent Bernoulli subsets; exact zeros/ones stay exact."""
    choices = [(False, True) if 0 < p < 1 else (bool(p),) for p in presence]
    result = []
    for selected in product(*choices):
        mask = np.asarray(selected, dtype=bool)
        probability = float(np.prod(np.where(mask, presence, 1 - presence)))
        result.append((np.flatnonzero(mask), probability))
    return result


def project_with_jacobian(
    point: FloatArray, matrices: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Pinhole projection and analytic quotient-rule Jacobian (no depth clamp)."""
    q = matrices[..., :3] @ point + matrices[..., 3]
    depth = q[:, 2]
    if (np.abs(depth) < 1e-10).any():
        raise ValueError("Projection lies on a camera plane")
    uv = q[:, :2] / depth[:, None]
    jac = (matrices[:, :2, :3] - uv[:, :, None] * matrices[:, 2, None, :3]) / depth[
        :, None, None
    ]
    return uv, jac, depth


def fit_component(
    matrices: FloatArray,
    means: FloatArray,
    covariance: FloatArray,
    prior: GaussianPrior3D,
    *,
    max_nfev: int,
) -> tuple[FloatArray, FloatArray, float]:
    """Return MAP, Gauss-Newton Laplace covariance and log evidence.

    A Gaussian spatial prior is mandatory, including for zero/one view.
    Numerical failure or a mode behind an active camera is an explicit error;
    no component is silently dropped, jittered, or replaced by a point estimate.
    """
    if len(matrices) == 0:
        return prior.mean.copy(), prior.covariance.copy(), 0.0
    whitening = np.linalg.inv(np.linalg.cholesky(covariance))
    prior_whitening = np.linalg.inv(np.linalg.cholesky(prior.covariance))

    def residual(point: FloatArray) -> FloatArray:
        uv, _, _ = project_with_jacobian(point, matrices)
        return np.concatenate(
            (
                prior_whitening @ (point - prior.mean),
                np.einsum("vij,vj->vi", whitening, uv - means).ravel(),
            )
        )

    def jacobian(point: FloatArray) -> FloatArray:
        _, jac, _ = project_with_jacobian(point, matrices)
        return np.concatenate((prior_whitening, (whitening @ jac).reshape(-1, 3)))

    fit = least_squares(
        residual,
        prior.mean,
        jac=jacobian,
        max_nfev=max_nfev,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f"Triangulation optimization failed: {fit.message}")
    _, _, depth = project_with_jacobian(fit.x, matrices)
    if (depth <= 0).any():
        raise RuntimeError("Triangulation mode is behind an active camera")
    jac = jacobian(fit.x)
    posterior_covariance = np.linalg.inv(jac.T @ jac)
    log_normalizer = 0.5 * (
        (3 + 2 * len(matrices)) * math.log(2 * math.pi)
        + float(np.linalg.slogdet(prior.covariance)[1])
        + float(np.linalg.slogdet(covariance)[1].sum())
    )
    log_evidence = (
        -0.5 * float(residual(fit.x) @ residual(fit.x))
        - log_normalizer
        + 1.5 * math.log(2 * math.pi)
        + 0.5 * float(np.linalg.slogdet(posterior_covariance)[1])
    )
    return np.asarray(fit.x, np.float64), posterior_covariance, log_evidence


def triangulate_gmm(
    observations: CameraGMM,
    cameras: tuple[PinholeCamera, ...],
    *,
    prior: GaussianPrior3D,
    config: LaplaceConfig,
) -> ProbabilisticTriangulation:
    """Method A, with within-subset evidence and explicit presence marginalization.

    Each camera subset S gets its supplied Bernoulli mass. Within S, normalize
    p0(x) times the product of projected 2D GMMs. Component evidence includes
    covariance determinants and the Laplace volume, not just reprojection error.
    This is a conditional fusion approximation, not a generative absence model.
    """
    v, _, _ = observations.means_px.shape
    if len(cameras) != v or len({c.camera_id for c in cameras}) != v:
        raise ValueError("Distinct cameras must match the GMM view axis")
    nonzero = [np.flatnonzero(w > 0) for w in observations.weights]
    count = math.prod(
        (len(nonzero[i]) if p > 0 else 0) + (1 if p < 1 else 0)
        for i, p in enumerate(observations.presence)
    )
    if count > config.max_components:
        raise ValueError(
            f"Exact enumeration requires {count} components, budget={config.max_components}"
        )
    subsets = camera_subsets(observations.presence)
    matrices = np.stack([c.matrix for c in cameras])
    means, covariances, weights, masks = [], [], [], []
    for active, probability in subsets:
        evidence = []
        for combination in product(*(nonzero[i] for i in active)):
            index = np.asarray(combination, dtype=np.int64)
            mean, cov, log_evidence = fit_component(
                matrices[active],
                observations.means_px[active, index],
                observations.covariance_px2[active, index],
                prior,
                max_nfev=config.max_nfev,
            )
            evidence.append(
                log_evidence + float(np.log(observations.weights[active, index]).sum())
            )
            means.append(mean)
            covariances.append(cov)
            mask = np.zeros(v, dtype=bool)
            mask[active] = True
            masks.append(mask)
        logs = np.asarray(evidence, dtype=np.float64)
        weights.extend(probability * np.exp(logs - logsumexp(logs)))
    return ProbabilisticTriangulation(
        GaussianMixture3D(np.stack(means), np.stack(covariances), np.asarray(weights)),
        np.stack(masks),
        float(np.prod(1 - observations.presence)),
    )
