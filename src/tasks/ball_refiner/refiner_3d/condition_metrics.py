"""NLL and full-mixture HDR volume in world metres, without mode selection."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from src.utils.geometry.probabilistic_triangulation.distributions import (
    FloatArray,
    GaussianMixture3D,
)

LEVELS = (0.5, 0.9, 0.95)


def mixture_hdr_metrics(mixture: GaussianMixture3D, truth: FloatArray, *, samples: int, seed: int) -> dict[str, FloatArray]:
    """Independent threshold/volume draws; MC SE is conditional on the threshold.

    HDR volume is E_q[1(log q(X)>=threshold)/q(X)] over R^3, not a
    moment ellipsoid or sum of component volumes. Zero-weight components stay.
    """
    if samples < 100 or truth.shape != (3,) or not np.isfinite(truth).all():
        raise ValueError('Need >=100 samples and a finite XYZ truth')
    precision = np.linalg.inv(mixture.covariance)
    with np.errstate(divide='ignore'):
        offset = np.log(mixture.weights) - .5 * (np.linalg.slogdet(mixture.covariance)[1] + 3 * np.log(2 * np.pi))

    def density(points: FloatArray) -> FloatArray:
        chunks = []
        for start in range(0, len(points), 512):
            delta = points[start:start + 512, None] - mixture.means
            quadratic = np.einsum('smi,mij,smj->sm', delta, precision, delta)
            chunks.append(logsumexp(offset - .5 * quadratic, axis=-1))
        return np.asarray(np.concatenate(chunks), dtype=np.float64)

    rng = np.random.default_rng(seed)
    threshold = np.quantile(density(mixture.sample(samples, rng)), 1 - np.asarray(LEVELS))
    log_volume = density(mixture.sample(samples, rng))
    volume_terms = np.where(log_volume[:, None] >= threshold, np.exp(-log_volume[:, None]), 0.)
    log_gt = density(truth[None])[0]
    result = {'nll_nat': np.asarray(-log_gt), 'covered': np.asarray(log_gt >= threshold, dtype=np.float64),
              'log_density_threshold': threshold, 'volume_m3': volume_terms.mean(0),
              'volume_mc_se_m3': volume_terms.std(0, ddof=1) / np.sqrt(samples)}
    if any(not np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError('Nonfinite mixture density/HDR metric')
    return result
