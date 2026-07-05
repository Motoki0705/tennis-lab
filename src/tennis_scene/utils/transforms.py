"""Coordinate transformations for tennis scene reconstruction.

The implementations now live in :mod:`src.utils.geometry`; this module
re-exports them to preserve the historical
``src.tennis_scene.utils.transforms`` import path.
"""

from __future__ import annotations

from src.utils.geometry.keypoints import denormalize_keypoints, normalize_keypoints
from src.utils.geometry.matrices import (
    apply_plcs_transform,
    apply_plcs_transform_batch,
    rotation_matrix_y,
    smpl_y_up_to_court_z_up,
)

__all__ = [
    "apply_plcs_transform",
    "apply_plcs_transform_batch",
    "denormalize_keypoints",
    "normalize_keypoints",
    "rotation_matrix_y",
    "smpl_y_up_to_court_z_up",
]
