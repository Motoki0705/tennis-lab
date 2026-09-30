"""One explicit conditioning policy for synthetic generation and later inference."""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np

from src.utils.geometry.triangulation import PinholeCamera

from .convergence import (
    CheckedTriangulation,
    ConvergenceConfig,
    RayConvergenceConfig,
    convergence_config,
    triangulate_converged,
)
from .distributions import CameraGMM, GaussianPrior3D
from .solver import HybridConfig, LaplaceConfig, triangulate_hybrid
from .volume import VoxelConfig

ConditioningConfig: TypeAlias = VoxelConfig | ConvergenceConfig | RayConvergenceConfig


def conditioning_config(values: dict[str, Any]) -> ConditioningConfig:
    """Fixed hybrid is explicit; historical convergence budgets retain meaning."""
    settings = dict(values)
    if settings.get("method") == "fixed_hybrid":
        settings.pop("method")
        return VoxelConfig(**settings)
    return convergence_config(settings)


def triangulate_conditioning(
    observations: CameraGMM, cameras: tuple[PinholeCamera, ...], *,
    prior: GaussianPrior3D, laplace: LaplaceConfig, config: ConditioningConfig,
) -> CheckedTriangulation:
    """Use the declared method once; numerical convergence is diagnostic only.

    A fixed budget does not estimate integration convergence. Its false flags
    and zero delta placeholders MUST be interpreted with convergence_assessed
    false, never as evidence of either convergence or divergence.
    """
    if not isinstance(config, VoxelConfig):
        return triangulate_converged(observations, cameras, prior=prior, laplace=laplace, config=config)
    posterior = triangulate_hybrid(observations, cameras, prior=prior, config=HybridConfig(laplace, config))
    count = len(posterior.distribution.weights)
    return CheckedTriangulation(posterior, False, 1, np.zeros(count, dtype=bool),
                                np.zeros((count, 3)), 0., (), convergence_assessed=False)
