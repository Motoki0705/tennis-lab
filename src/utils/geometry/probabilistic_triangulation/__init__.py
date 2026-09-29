"""Source-pixel GMMs to a world-space mixture; see README for approximations."""

from .distributions import CameraGMM, GaussianMixture3D, GaussianPrior3D
from .solver import LaplaceConfig, ProbabilisticTriangulation, triangulate_gmm

__all__ = [
    "CameraGMM",
    "GaussianMixture3D",
    "GaussianPrior3D",
    "LaplaceConfig",
    "ProbabilisticTriangulation",
    "triangulate_gmm",
]
