"""Human skeleton renderer for 2D and 3D visualization.

This module re-exports from src.utils.rendering.skeleton_renderer for backward compatibility.

Note:
    New code should import from src.utils.rendering directly.

"""

from src.utils.rendering.skeleton_renderer import (
    COCO17_KEYPOINT_NAMES,
    COCO17_SKELETON,
    SKELETON_DEFINITIONS,
    SMPL_SKELETON,
    SMPLH_SKELETON,
    SkeletonRenderer,
    SkeletonStyle,
    SkeletonType,
)

__all__ = [
    "SkeletonRenderer",
    "SkeletonStyle",
    "SkeletonType",
    "COCO17_SKELETON",
    "COCO17_KEYPOINT_NAMES",
    "SMPLH_SKELETON",
    "SMPL_SKELETON",
    "SKELETON_DEFINITIONS",
]
