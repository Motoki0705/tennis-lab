"""CPU research baselines B/C. These are not production dispatch alternatives."""

from __future__ import annotations

from itertools import product

import numpy as np
from numpy.typing import NDArray

from src.utils.geometry.probabilistic_triangulation import (
    CameraGMM,
    GaussianMixture3D,
    GaussianPrior3D,
)
from src.utils.geometry.probabilistic_triangulation.distributions import camera_subsets
from src.utils.geometry.probabilistic_triangulation.solver import (
    ProbabilisticTriangulation,
    fit_component,
    fit_component_approximate,
)
from src.utils.geometry.probabilistic_triangulation.volume import (
    VoxelConfig as VoxelConfig,
)
from src.utils.geometry.probabilistic_triangulation.volume import (
    triangulate_volume as triangulate_volume,
)
from src.utils.geometry.triangulation import PinholeCamera


def triangulate_samples(
    observations: CameraGMM,
    cameras: tuple[PinholeCamera, ...],
    *,
    prior: GaussianPrior3D,
    samples: int,
    rng: np.random.Generator,
    max_nfev: int,
) -> GaussianMixture3D:
    """Method C: sample 2D GMMs and randomize-then-optimize triangulation.

    Perturb the spatial prior as well (otherwise depth variance is lost in the
    zero/one-view case). Equal particles do NOT correct incompatible component
    combinations by cross-view evidence. Scott KDE supplies a measurable density;
    its bandwidth is part of this baseline, not a claimed exact posterior.
    """
    if samples < 8:
        raise ValueError("At least 8 particles required for 3D KDE")
    matrices = np.stack([c.matrix for c in cameras])
    points = []
    for _ in range(samples):
        active = np.flatnonzero(rng.random(len(cameras)) < observations.presence)
        selected = np.asarray(
            [
                rng.choice(len(observations.weights[i]), p=observations.weights[i])
                for i in active
            ],
            dtype=np.int64,
        )
        means = observations.means_px[active, selected]
        cov = observations.covariance_px2[active, selected]
        pixels = means + np.einsum(
            "vij,vj->vi", np.linalg.cholesky(cov), rng.standard_normal((len(active), 2))
        )
        perturbed = GaussianPrior3D(
            rng.multivariate_normal(prior.mean, prior.covariance), prior.covariance
        )
        point, _, _ = fit_component(
            matrices[active], pixels, cov, perturbed, max_nfev=max_nfev
        )
        points.append(point)
    particles = np.stack(points)
    bandwidth = np.cov(particles.T) * samples ** (-2 / 7)
    return GaussianMixture3D(
        particles,
        np.broadcast_to(bandwidth, (samples, 3, 3)).copy(),
        np.full(samples, 1 / samples),
    )


def triangulate_stratified_samples(
    observations: CameraGMM, cameras: tuple[PinholeCamera, ...], *,
    prior: GaussianPrior3D, samples_per_product: int, seed: int, max_nfev: int,
) -> ProbabilisticTriangulation:
    """Fixed-seed C, retaining every positive-mass subset/component product.

    Perturb 2D means and prior centers independently, then fit in positive depth.
    Moment-match each product's particles, with the Scott KDE within-particle
    covariance. Product weights are the supplied 2D weights (no evidence
    correction); empty subsets are exactly the prior. This is an explicitly
    approximate research comparator, not a Monte Carlo posterior estimator.
    """
    if samples_per_product < 8 or max_nfev < 1:
        raise ValueError("Need >=8 samples per product and a positive fitting budget")
    if len(cameras) != len(observations.presence) or len({c.camera_id for c in cameras}) != len(cameras):
        raise ValueError("Distinct cameras must match the GMM view axis")
    rng = np.random.default_rng(seed)
    matrices = np.stack([c.matrix for c in cameras])
    nonzero = [np.flatnonzero(w > 0) for w in observations.weights]
    means, covariances, weights, masks, methods = [], [], [], [], []
    diagnostics: list[dict[str, float | int | bool]] = []
    for active, probability in camera_subsets(observations.presence):
        for combination in product(*(nonzero[i] for i in active)):
            index = np.asarray(combination, dtype=np.int64)
            mask: NDArray[np.bool_] = np.zeros(len(cameras), dtype=bool)
            mask[active] = True
            masks.append(mask)
            weights.append(probability * float(np.prod(observations.weights[active, index])))
            if not len(active):
                means.append(prior.mean.copy())
                covariances.append(prior.covariance.copy())
                methods.append("prior")
                diagnostics.append({"optimizer_converged": True, "nonregular_samples": 0})
                continue
            covariance = observations.covariance_px2[active, index]
            particles, fits = [], []
            for _ in range(samples_per_product):
                pixels = observations.means_px[active, index] + np.einsum(
                    "vij,vj->vi", np.linalg.cholesky(covariance), rng.standard_normal((len(active), 2)),
                )
                perturbed = GaussianPrior3D(rng.multivariate_normal(prior.mean, prior.covariance), prior.covariance)
                fit = fit_component_approximate(matrices[active], pixels, covariance, perturbed, max_nfev=max_nfev)
                particles.append(fit.mean)
                fits.append(fit)
            points = np.stack(particles)
            means.append(points.mean(0))
            # Empirical particle population covariance + each KDE kernel's
            # Scott bandwidth. No diagonal jitter or rank-deficiency repair.
            empirical = np.cov(points.T)
            covariances.append(empirical * ((samples_per_product - 1) / samples_per_product + samples_per_product ** (-2 / 7)))
            methods.append("samples")
            diagnostics.append({"optimizer_converged": all(f.diagnostics["optimizer_converged"] for f in fits),
                                "nonregular_samples": sum(f.reason is not None for f in fits)})
    return ProbabilisticTriangulation(
        GaussianMixture3D(np.stack(means), np.stack(covariances), np.asarray(weights)),
        np.stack(masks), float(np.prod(1 - observations.presence)), tuple(methods),
        np.zeros(len(weights)), component_optimization_diagnostics=tuple(diagnostics),
    )
