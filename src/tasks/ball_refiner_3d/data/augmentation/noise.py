"""Pixel noise applied before triangulation."""

from __future__ import annotations

from typing import cast

import numpy as np

from src.tasks.ball_refiner_3d.configuration.data import CorruptionConfig
from src.tasks.ball_refiner_3d.data.schema import FloatArray


def coordinate_noise(
    shape: tuple[int, ...], config: CorruptionConfig, rng: np.random.Generator
) -> FloatArray:
    """Isotropic small jitter + Gaussian outliers with population radial P95.

    The Rayleigh tail at R is exp(-R²/2σ²). Solve the two-component survival
    function exactly for the broad component at R=config.noise_p95_px.
    """
    radius = config.noise_p95_px
    if radius == 0:
        return np.zeros((*shape, 2), dtype=np.float32)
    small_tail = (
        np.exp(-0.5 * (radius / config.jitter_sigma_px) ** 2)
        if config.jitter_sigma_px
        else 0.0
    )
    broad_tail = (
        0.05 - (1 - config.outlier_probability) * small_tail
    ) / config.outlier_probability
    broad_sigma = radius / np.sqrt(-2 * np.log(broad_tail))
    outlier = rng.random(shape) < config.outlier_probability
    sigma = np.where(outlier, broad_sigma, config.jitter_sigma_px)
    return cast(
        FloatArray,
        (rng.standard_normal((*shape, 2)) * sigma[..., None]).astype(np.float32),
    )
