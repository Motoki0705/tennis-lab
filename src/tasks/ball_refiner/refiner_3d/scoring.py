"""GT scoring of full 3D mixtures; never used to fit or select input frames."""

from __future__ import annotations

import numpy as np

from src.utils.geometry.probabilistic_triangulation.distributions import (
    FloatArray,
    GaussianMixture3D,
)


def score_mixture(
    mixture: GaussianMixture3D, truth: FloatArray, *, samples: int, seed: int,
) -> dict[str, float | bool]:
    """NLL (nat/world-unit³), L2 error, and Monte Carlo mixture HDR coverage.

    For mass a, the HDR is {x: log q(x) >= quantile_(1-a)(log q(X))}, X~q.
    Draws use a fixed seed, independently of GT. Chunking limits memory without
    changing the draws or thresholds. Neither moment ellipsoids nor per-mode
    intervals are substituted for the mixture HDR.
    """
    if truth.shape != (3,) or samples < 100:
        raise ValueError("Need one 3D GT and at least 100 HDR samples")
    log_gt = float(mixture.log_prob(truth))
    draws = mixture.sample(samples, np.random.default_rng(seed))
    density = np.concatenate([mixture.log_prob(draws[i:i + 512]) for i in range(0, samples, 512)])
    result: dict[str, float | bool] = {
        "nll_nat": -log_gt,
        "mean_error_m": float(np.linalg.norm(mixture.moments()[0] - truth)),
    }
    for mass in (50, 90, 95):
        threshold = float(np.quantile(density, 1 - mass / 100))
        result[f"hdr{mass}"] = log_gt >= threshold
        result[f"hdr{mass}_log_density_threshold"] = threshold
    return result
