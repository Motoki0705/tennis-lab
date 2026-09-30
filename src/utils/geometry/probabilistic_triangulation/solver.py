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

from .adaptive import AdaptiveRayConfig, AdaptiveRayProposal
from .optimization import NonregularComponentError, feasible_start
from .ray import RayConfig, RayProposal, single_view_moments
from .volume import VoxelConfig, integrate_component

COMPONENT_METHODS: tuple[str, ...] = (
    "prior", "laplace", "volume:camera_boundary", "volume:boundary_laplace_tail",
    "volume:iteration_budget", "volume:line_search", "volume:no_feasible_initial_point",
)


@dataclass(frozen=True)
class LaplaceConfig:
    max_components: int
    max_nfev: int
    diagnose_nonregular: bool = False

    def __post_init__(self) -> None:
        if self.max_components < 1 or self.max_nfev < 1:
            raise ValueError("Component and optimizer budgets must be positive")


@dataclass(frozen=True)
class ProbabilisticTriangulation:
    distribution: GaussianMixture3D
    camera_subsets: NDArray[np.bool_]  # M,V; same order as mixture components
    prior_only_probability: float
    component_methods: tuple[str, ...]
    component_log_evidence: FloatArray
    component_integration_diagnostics: tuple[dict[str, float | int | bool], ...] = ()
    component_optimization_diagnostics: tuple[dict[str, float | int | bool], ...] = ()


COMPONENT_METHODS += tuple(method.replace("volume:", "ray:") for method in COMPONENT_METHODS if method.startswith("volume:"))
COMPONENT_METHODS += tuple(method.replace("volume:", "adaptive_ray:") for method in COMPONENT_METHODS if method.startswith("volume:"))
COMPONENT_METHODS += tuple(method.replace("volume:", "laplace_diagnostic:") for method in COMPONENT_METHODS if method.startswith("volume:") and not method.endswith("no_feasible_initial_point"))


ComponentCache = dict[tuple[int, ...], tuple[FloatArray, FloatArray, float] | str]



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


@dataclass(frozen=True)
class ComponentFit:
    mean: FloatArray
    covariance: FloatArray
    log_evidence: float
    reason: str | None
    diagnostics: dict[str, float | int | bool]


def _fit_component(
    matrices: FloatArray,
    means: FloatArray,
    covariance: FloatArray,
    prior: GaussianPrior3D,
    *,
    max_nfev: int,
    strict: bool,
) -> ComponentFit:
    """Return the last positive-depth Gauss-Newton approximation and diagnostics.

    A Gaussian spatial prior is mandatory, including for zero/one view.
    Boundary/tail/iteration diagnostics do not change this deterministic path.
    A budget-limited point is not claimed to be a MAP. Invalid arithmetic, an
    infeasible camera intersection, or a non-SPD covariance still raises.
    """
    if len(matrices) == 0:
        return ComponentFit(prior.mean.copy(), prior.covariance.copy(), 0.0, None,
                            {"optimizer_converged": True, "evaluations": 0})
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
    reason: str | None = None
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
            reason = "camera_boundary"
            break
        for _ in range(40):
            candidate = point + fraction * step
            if evaluations >= max_nfev:
                reason = "iteration_budget"
                break
            trial = residual(candidate)
            evaluations += 1
            trial_cost = .5 * float(trial @ trial)
            if trial_cost <= cost + 1e-4 * fraction * float((jac.T @ r) @ step):
                break
            fraction *= .5
        else:
            reason = "line_search"
        if reason is not None:
            break
        point, r = candidate, trial
        if abs(cost - trial_cost) < 1e-12 * (1 + cost):
            converged = True
            break
    if not converged:
        reason = reason or "iteration_budget"
    if strict and reason is not None:
        raise NonregularComponentError(reason)
    _, _, depth = project_with_jacobian(point, matrices)
    if (depth <= minimum_depth).any():
        raise NonregularComponentError("camera_boundary")
    jac = jacobian(point)
    posterior_covariance = np.linalg.inv(jac.T @ jac)
    depth_sigma = np.sqrt(np.einsum('vi,ij,vj->v', matrices[:, 2, :3], posterior_covariance, matrices[:, 2, :3]))
    if (depth < 3 * depth_sigma).any():
        reason = reason or "boundary_laplace_tail"
    if strict and reason is not None:
        raise NonregularComponentError(reason)
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
    return ComponentFit(np.asarray(point, np.float64), posterior_covariance, log_evidence,
                        reason, {"optimizer_converged": converged, "evaluations": evaluations,
                                 "minimum_depth": float(depth.min()),
                                 "minimum_depth_sigmas": float((depth / depth_sigma).min())})


def fit_component_approximate(
    matrices: FloatArray, means: FloatArray, covariance: FloatArray,
    prior: GaussianPrior3D, *, max_nfev: int,
) -> ComponentFit:
    """Explicit diagnostic policy: keep a positive-depth approximation and flags."""
    return _fit_component(matrices, means, covariance, prior, max_nfev=max_nfev, strict=False)


def fit_component(
    matrices: FloatArray, means: FloatArray, covariance: FloatArray,
    prior: GaussianPrior3D, *, max_nfev: int,
) -> tuple[FloatArray, FloatArray, float]:
    """Strict regular-interior policy used by the historical hybrid methods."""
    fit = _fit_component(matrices, means, covariance, prior, max_nfev=max_nfev, strict=True)
    return fit.mean, fit.covariance, fit.log_evidence


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
    prior: GaussianPrior3D, config: LaplaceConfig, volume: VoxelConfig | RayConfig | None,
    regular_cache: ComponentCache | None = None,
    ray_cache: dict[tuple[int, ...], RayProposal | AdaptiveRayProposal] | None = None,
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
    methods, component_evidence, diagnostics = [], [], []
    optimizer_diagnostics = []
    for active, probability in subsets:
        evidence = []
        for combination in product(*(nonzero[i] for i in active)):
            index = np.asarray(combination, dtype=np.int64)
            method = "laplace" if len(active) else "prior"
            diagnostic: dict[str, float | int | bool] = {}
            optimizer_diagnostic: dict[str, float | int | bool] = {}
            key = tuple(int(index[list(active).index(i)]) if i in active else -1 for i in range(v))
            try:
                cached = regular_cache.get(key) if regular_cache is not None else None
                if isinstance(cached, str):
                    raise NonregularComponentError(cached)
                if config.diagnose_nonregular:
                    if volume is not None or regular_cache is not None:
                        raise ValueError("Diagnostic Laplace is a single path, not a hybrid")
                    fit = fit_component_approximate(
                        matrices[active], observations.means_px[active, index],
                        observations.covariance_px2[active, index], prior, max_nfev=config.max_nfev,
                    )
                    cached = fit.mean, fit.covariance, fit.log_evidence
                    optimizer_diagnostic = fit.diagnostics
                    if fit.reason is not None:
                        method = f"laplace_diagnostic:{fit.reason}"
                elif cached is None:
                    cached = fit_component(
                        matrices[active], observations.means_px[active, index],
                        observations.covariance_px2[active, index], prior,
                        max_nfev=config.max_nfev,
                    )
                    if regular_cache is not None:
                        regular_cache[key] = cached
                mean, cov, log_evidence = cached
            except NonregularComponentError as exc:
                if regular_cache is not None:
                    regular_cache[key] = exc.reason
                if volume is None:
                    raise
                selected_cameras = tuple(cameras[i] for i in active)
                selected_means = observations.means_px[active, index]
                selected_covariance = observations.covariance_px2[active, index]
                if isinstance(volume, RayConfig):
                    method = f"ray:{exc.reason}"
                    if len(active) == 1:
                        mean, cov, log_evidence = single_view_moments(selected_cameras[0], selected_means[0], selected_covariance[0], prior, volume.order)
                    else:
                        proposal = ray_cache.get(key) if ray_cache is not None else None
                        if proposal is None:
                            proposal = RayProposal(selected_cameras, selected_means, selected_covariance, prior, adaptive_metric=isinstance(volume, AdaptiveRayConfig))
                            if isinstance(volume, AdaptiveRayConfig):
                                proposal = AdaptiveRayProposal(proposal)
                            if ray_cache is not None:
                                ray_cache[key] = proposal
                        if isinstance(proposal, AdaptiveRayProposal):
                            if not isinstance(volume, AdaptiveRayConfig):
                                raise TypeError("Adaptive cache requires adaptive configuration") from exc
                            mean, cov, log_evidence = proposal.integrate(volume)
                            method = f"adaptive_ray:{exc.reason}"
                            diagnostic = dict(proposal.diagnostic)
                        else:
                            mean, cov, log_evidence = proposal.integrate(volume.order)
                else:
                    method = f"volume:{exc.reason}"
                    mean, cov, log_evidence = integrate_component(selected_cameras, selected_means, selected_covariance, prior, volume)
            methods.append(method)
            diagnostics.append(diagnostic)
            optimizer_diagnostics.append(optimizer_diagnostic)
            component_evidence.append(log_evidence)
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
        np.asarray(component_evidence, dtype=np.float64),
        tuple(diagnostics),
        tuple(optimizer_diagnostics),
    )
