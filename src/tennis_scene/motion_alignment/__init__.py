"""Motion-to-motion alignment primitives for :mod:`src.tennis_scene`.

This package owns the geometric cores used to place an external motion track
(for example GVHMR world motion) into the court frame. Callers import the
public names from their owning module; this package re-exports the similarity
core for convenience.
"""

from src.tennis_scene.motion_alignment.similarity import (
    FIXED_SCALE,
    HEADING_CIRCULAR_MEAN,
    POSITION_PROJECTION_RATIO,
    POSITION_SVD_UMEYAMA,
    POSITION_SVD_XY_SCALE,
    SimilarityConfig,
    SimilarityFitDiagnostics,
    SimilarityFitResult,
    SimilarityTransform,
    fit_similarity,
    wrap_to_pi,
)

__all__ = [
    "FIXED_SCALE",
    "HEADING_CIRCULAR_MEAN",
    "POSITION_PROJECTION_RATIO",
    "POSITION_SVD_UMEYAMA",
    "POSITION_SVD_XY_SCALE",
    "SimilarityConfig",
    "SimilarityFitDiagnostics",
    "SimilarityFitResult",
    "SimilarityTransform",
    "fit_similarity",
    "wrap_to_pi",
]
