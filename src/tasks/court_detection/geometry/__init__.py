"""Geometry helpers for court-detection keypoints."""

from src.tasks.court_detection.geometry.confidence_homography import (
    ConfidenceHomographyResult,
    estimate_confidence_homography,
)
from src.tasks.court_detection.geometry.homography import (
    compute_template_to_image_homography,
    court_template_xy,
    estimate_homography,
    project_points,
)
from src.tasks.court_detection.geometry.hybrid_homography import (
    HybridHomographyConfig,
    HybridHomographyResult,
    estimate_hybrid_homography,
)
from src.tasks.court_detection.geometry.postprocess import (
    HomographyPostprocessResult,
    refine_court_keypoints_with_homography,
)

__all__ = [
    "ConfidenceHomographyResult",
    "HomographyPostprocessResult",
    "HybridHomographyConfig",
    "HybridHomographyResult",
    "compute_template_to_image_homography",
    "court_template_xy",
    "estimate_homography",
    "estimate_confidence_homography",
    "estimate_hybrid_homography",
    "project_points",
    "refine_court_keypoints_with_homography",
]
