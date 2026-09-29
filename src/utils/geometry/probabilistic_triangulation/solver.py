"""Enumerated component products with covariance-weighted nonlinear fitting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from src.utils.geometry.probabilistic_triangulation.distributions import (
    CameraGMM,
    FloatArray,
    GaussianMixture3D,
    GaussianPrior3D,
    camera_subsets,
)
from src.utils.geometry.triangulation import PinholeCamera

from .optimization import NonregularComponentError, feasible_start
from .volume import VoxelConfig, integrate_component

COMPONENT_METHODS = (
    "prior", "laplace", "volume:camera_boundary", "volume:boundary_laplace_tail",
    "volume:iteration_budget", "volume:line_search", "volume:no_feasible_initial_point",
)


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
    component_methods: tuple[str, ...]



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
    All iterates stay in front of active cameras; a nonregular mode is an error;
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

    # Every accepted/trial iterate stays in the positive-depth convex region.
    minimum_depth = 1e-4
    point = feasible_start(prior.mean, matrices, minimum_depth)
    r = residual(point)
    evaluations = 1
    converged = False
    while evaluations < max_nfev:
        jac = jacobian(point)
        precision = jac.T @ jac
        step = -np.linalg.solve(precision, jac.T @ r)
        cost = .5 * float(r @ r)
        if float(-step @ (jac.T @ r)) < 1e-12 * (1 + cost):
            converged = True
            break
        depth = matrices[:, 2, :3] @ point + matrices[:, 2, 3]
        change = matrices[:, 2, :3] @ step
        approaching = change < 0
        fraction = min(1., float(np.min(.99 * (depth[approaching] - minimum_depth) / -change[approaching]))) if approaching.any() else 1.
        if float(depth.min()) < 10 * minimum_depth:
            raise NonregularComponentError("camera_boundary")
        for _ in range(40):
            candidate = point + fraction * step
            if evaluations >= max_nfev:
                raise NonregularComponentError('iteration_budget')
            trial = residual(candidate)
            evaluations += 1
            trial_cost = .5 * float(trial @ trial)
            if trial_cost <= cost + 1e-4 * fraction * float((jac.T @ r) @ step):
                break
            fraction *= .5
        else:
            raise NonregularComponentError("line_search")
        point, r = candidate, trial
        if abs(cost - trial_cost) < 1e-12 * (1 + cost):
            converged = True
            break
    if not converged:
        raise NonregularComponentError("iteration_budget")
    _, _, depth = project_with_jacobian(point, matrices)
    if (depth <= minimum_depth).any():
        raise NonregularComponentError("camera_boundary")
    jac = jacobian(point)
    posterior_covariance = np.linalg.inv(jac.T @ jac)
    depth_sigma = np.sqrt(np.einsum('vi,ij,vj->v', matrices[:, 2, :3], posterior_covariance, matrices[:, 2, :3]))
    if (depth < 3 * depth_sigma).any():
        raise NonregularComponentError('boundary_laplace_tail')
    log_normalizer = 0.5 * (
        (3 + 2 * len(matrices)) * math.log(2 * math.pi)
        + float(np.linalg.slogdet(prior.covariance)[1])
        + float(np.linalg.slogdet(covariance)[1].sum())
    )
    log_evidence = (
        -0.5 * float(residual(point) @ residual(point))
        - log_normalizer
        + 1.5 * math.log(2 * math.pi)
        + 0.5 * float(np.linalg.slogdet(posterior_covariance)[1])
    )
    return np.asarray(point, np.float64), posterior_covariance, log_evidence


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
    return _triangulate(observations, cameras, prior=prior, config=config, volume=None)


@dataclass(frozen=True)
class HybridConfig:
    laplace: LaplaceConfig
    volume: VoxelConfig


def triangulate_hybrid(
    observations: CameraGMM, cameras: tuple[PinholeCamera, ...], *,
    prior: GaussianPrior3D, config: HybridConfig,
) -> ProbabilisticTriangulation:
    """Explicit A/B dispatch per product, with mandatory component diagnostics.

    Regular positive-depth modes use A. Boundary and nonconverged products use
    the configured volume integral for evidence/moments. Every product remains
    present, and any volume failure aborts the frame.
    """
    return _triangulate(observations, cameras, prior=prior, config=config.laplace, volume=config.volume)


def _triangulate(
    observations: CameraGMM, cameras: tuple[PinholeCamera, ...], *,
    prior: GaussianPrior3D, config: LaplaceConfig, volume: VoxelConfig | None,
) -> ProbabilisticTriangulation:
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
    methods = []
    for active, probability in subsets:
        evidence = []
        for combination in product(*(nonzero[i] for i in active)):
            index = np.asarray(combination, dtype=np.int64)
            method = "laplace" if len(active) else "prior"
            try:
                mean, cov, log_evidence = fit_component(
                    matrices[active],
                    observations.means_px[active, index],
                    observations.covariance_px2[active, index],
                    prior,
                    max_nfev=config.max_nfev,
                )
            except NonregularComponentError as exc:
                if volume is None:
                    raise
                method = f"volume:{exc.reason}"
                mean, cov, log_evidence = integrate_component(
                    tuple(cameras[i] for i in active), observations.means_px[active, index],
                    observations.covariance_px2[active, index], prior, volume,
                )
            methods.append(method)
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
        tuple(methods),
    )
