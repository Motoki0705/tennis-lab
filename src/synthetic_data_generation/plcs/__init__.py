"""3DGS-native PLCS avatar construction and motion-control contracts."""

from src.synthetic_data_generation.plcs.avatar_asset import (
    AvatarGaussianAsset,
    DeformedAvatarGaussians,
    build_surface_gaussian_asset,
    deform_avatar_gaussians,
)
from src.synthetic_data_generation.plcs.avatar_control import (
    NeighborBlend,
    apply_joint_linear_blend_skinning,
    apply_vertex_transform_blend,
    embed_points_on_posed_mesh,
    hugs_topk_neighbor_blend,
    interpolate_face_attributes,
)
from src.synthetic_data_generation.plcs.planner import (
    POSE_IDS,
    PLCSPersonSchedule,
    build_prototype_person_schedule,
)

__all__ = [
    "NeighborBlend",
    "POSE_IDS",
    "PLCSPersonSchedule",
    "AvatarGaussianAsset",
    "DeformedAvatarGaussians",
    "apply_joint_linear_blend_skinning",
    "apply_vertex_transform_blend",
    "embed_points_on_posed_mesh",
    "hugs_topk_neighbor_blend",
    "interpolate_face_attributes",
    "build_surface_gaussian_asset",
    "build_prototype_person_schedule",
    "deform_avatar_gaussians",
]
