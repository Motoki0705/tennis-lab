"""Shared geometry utilities.

Submodules:

- :mod:`angles` — wrapped-angle / vector primitives (torch).
- :mod:`skeleton` — COCO-17 joint-angle, torsion, twist, bone-length (torch).
- :mod:`court_pose` — normalized-court <-> world/court pose conversions (torch).
- :mod:`matrices` — rotation matrices and rigid SMPL transforms (numpy).
- :mod:`keypoints` — pixel <-> normalized keypoint conversions (numpy).

Import from the relevant submodule (e.g.
``from src.utils.geometry.angles import angular_error``). The most common
symbols are also re-exported here for convenience.
"""

from src.utils.geometry.angles import (
    angular_error,
    normalize_vector,
    signed_angle_around_axis,
    wrapped_angle_diff,
)
from src.utils.geometry.court_pose import (
    canonical_pose_to_world_pose,
    court_position_to_world_translation,
    world_pose_to_canonical_pose,
)
from src.utils.geometry.keypoints import (
    clamp_pixel_coordinate,
    denormalize_keypoints,
    normalize_keypoints,
)
from src.utils.geometry.matrices import (
    apply_plcs_transform,
    apply_plcs_transform_batch,
    axis_angle_to_rotation_matrix,
    rotation_matrix_y,
    rotation_matrix_z,
)
from src.utils.geometry.skeleton import (
    compute_bone_lengths,
    compute_joint_angles,
    compute_torsion_angles,
    compute_torso_twist,
)

__all__ = [
    "angular_error",
    "apply_plcs_transform",
    "apply_plcs_transform_batch",
    "axis_angle_to_rotation_matrix",
    "canonical_pose_to_world_pose",
    "clamp_pixel_coordinate",
    "compute_bone_lengths",
    "compute_joint_angles",
    "compute_torsion_angles",
    "compute_torso_twist",
    "court_position_to_world_translation",
    "denormalize_keypoints",
    "normalize_keypoints",
    "normalize_vector",
    "rotation_matrix_y",
    "rotation_matrix_z",
    "signed_angle_around_axis",
    "wrapped_angle_diff",
    "world_pose_to_canonical_pose",
]
