"""CPU research baselines B/C. These are not production dispatch alternatives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from src.utils.geometry.probabilistic_triangulation import (
    CameraGMM,
    GaussianMixture3D,
    GaussianPrior3D,
)
from src.utils.geometry.probabilistic_triangulation.distributions import FloatArray
from src.utils.geometry.probabilistic_triangulation.solver import (
    camera_subsets,
    fit_component,
)
from src.utils.geometry.triangulation import PinholeCamera


@dataclass(frozen=True)
class VoxelConfig:
    initial_cells: int
    levels: int
    refine_cells: int
    prior_sigmas: float

    def __post_init__(self) -> None:
        if (
            min(self.initial_cells, self.levels, self.refine_cells) < 1
            or not math.isfinite(self.prior_sigmas)
            or self.prior_sigmas <= 0
        ):
            raise ValueError("Invalid voxel budget")


@dataclass(frozen=True)
class VoxelDensity:
    lower: FloatArray
    extent: FloatArray
    initial_cells: int
    centers: FloatArray
    widths: FloatArray
    weights: FloatArray
    leaves: dict[tuple[int, int, int, int], float]
    levels: int

    def log_prob(self, points: FloatArray) -> FloatArray:
        flat = points.reshape(-1, 3)
        output: FloatArray = np.full(len(flat), -np.inf, dtype=np.float64)
        for n, point in enumerate(flat):
            unit = (point - self.lower) / self.extent
            if (unit < 0).any() or (unit >= 1).any():
                continue
            for level in range(self.levels):
                x, y, z = np.floor(unit * (self.initial_cells * 2**level)).astype(int)
                value = self.leaves.get((level, int(x), int(y), int(z)))
                if value is not None:
                    output[n] = value
                    break
            else:
                raise RuntimeError("Adaptive voxel tree has a hole")
        return output.reshape(points.shape[:-1])

    def sample(self, count: int, rng: np.random.Generator) -> FloatArray:
        index = rng.choice(len(self.weights), count, p=self.weights)
        return self.centers[index] + (rng.random((count, 3)) - 0.5) * self.widths[index]


@dataclass(frozen=True)
class VolumeMixture:
    components: tuple[VoxelDensity, ...]
    weights: FloatArray

    def log_prob(self, points: FloatArray) -> FloatArray:
        return np.asarray(
            logsumexp(
                np.stack(
                    [
                        d.log_prob(points) + np.log(w)
                        for d, w in zip(self.components, self.weights, strict=True)
                    ]
                ),
                axis=0,
            ),
            np.float64,
        )

    def sample(self, count: int, rng: np.random.Generator) -> FloatArray:
        index = rng.choice(len(self.weights), count, p=self.weights)
        points = np.empty((count, 3))
        for i, density in enumerate(self.components):
            selected = index == i
            points[selected] = density.sample(int(selected.sum()), rng)
        return points


def _log_target(
    points: FloatArray,
    observations: CameraGMM,
    cameras: tuple[PinholeCamera, ...],
    active: list[int],
    prior: GaussianPrior3D,
) -> FloatArray:
    delta = points - prior.mean
    result = -0.5 * np.einsum(
        "ni,ij,nj->n", delta, np.linalg.inv(prior.covariance), delta
    )
    for view in active:
        uv, front = cameras[view].project(points)
        offset = uv[:, None] - observations.means_px[view]
        covariance = observations.covariance_px2[view]
        quadratic = np.einsum(
            "nki,kij,nkj->nk", offset, np.linalg.inv(covariance), offset
        )
        with np.errstate(divide="ignore"):
            log_component = np.log(observations.weights[view]) - 0.5 * (
                quadratic + np.linalg.slogdet(covariance)[1] + 2 * np.log(2 * np.pi)
            )
        result += np.where(front, logsumexp(log_component, axis=-1), -np.inf)
    return result


def _voxel_subset(
    observations: CameraGMM,
    cameras: tuple[PinholeCamera, ...],
    active: list[int],
    prior: GaussianPrior3D,
    config: VoxelConfig,
) -> VoxelDensity:
    # Axis-aligned finite box, no mode seeds/GT from method A. Unrefined cells
    # retain mass; refinement never silently discards the tails or other modes.
    extent = 2 * config.prior_sigmas * np.sqrt(prior.covariance.diagonal())
    lower = prior.mean - extent / 2
    indices = np.asarray(
        list(product(range(config.initial_cells), repeat=3)), dtype=np.int64
    )
    offsets = np.asarray(list(product(range(2), repeat=3)), dtype=np.int64)
    saved_indices, saved_levels, saved_log_mass = [], [], []
    for level in range(config.levels):
        width = extent / (config.initial_cells * 2**level)
        centers = lower + (indices + 0.5) * width
        log_mass = (
            _log_target(centers, observations, cameras, active, prior)
            + np.log(width).sum()
        )
        refine: NDArray[np.bool_] = np.zeros(len(indices), dtype=bool)
        if level < config.levels - 1:
            count = min(config.refine_cells, len(indices))
            refine[np.argsort(log_mass)[-count:]] = True
        saved_indices.append(indices[~refine])
        saved_levels.append(np.full((~refine).sum(), level, dtype=np.int64))
        saved_log_mass.append(log_mass[~refine])
        indices = (indices[refine, None] * 2 + offsets).reshape(-1, 3)
    all_indices = np.concatenate(saved_indices)
    levels = np.concatenate(saved_levels)
    log_mass = np.concatenate(saved_log_mass)
    normalizer = float(logsumexp(log_mass))
    if not math.isfinite(normalizer):
        raise RuntimeError("Voxel grid contains no finite likelihood mass")
    widths = extent / (config.initial_cells * 2.0 ** levels[:, None])
    centers = lower + (all_indices + 0.5) * widths
    log_density = log_mass - normalizer - np.log(widths).sum(1)
    leaves = {
        (int(level), int(ix[0]), int(ix[1]), int(ix[2])): float(log_p)
        for level, ix, log_p in zip(levels, all_indices, log_density, strict=True)
    }
    return VoxelDensity(
        lower,
        extent,
        config.initial_cells,
        centers,
        widths,
        np.exp(log_mass - normalizer),
        leaves,
        config.levels,
    )


def triangulate_volume(
    observations: CameraGMM,
    cameras: tuple[PinholeCamera, ...],
    *,
    prior: GaussianPrior3D,
    config: VoxelConfig,
) -> VolumeMixture:
    subsets = camera_subsets(observations.presence)
    return VolumeMixture(
        tuple(
            _voxel_subset(observations, cameras, list(active), prior, config)
            for active, _ in subsets
        ),
        np.asarray([p for _, p in subsets]),
    )


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
