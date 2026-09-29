"""CPU research baselines B/C. These are not production dispatch alternatives."""

from __future__ import annotations

import numpy as np

from src.utils.geometry.probabilistic_triangulation import (
    CameraGMM,
    GaussianMixture3D,
    GaussianPrior3D,
)
from src.utils.geometry.probabilistic_triangulation.solver import (
    fit_component,
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
