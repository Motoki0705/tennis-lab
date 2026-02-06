"""Target-generation utilities for PLCS datasets."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.utils.geometry import FACE_KEYPOINT_OFFSETS, SMPLH_TO_COCO17_MAPPING
from src.utils.geometry.constants import (
    COURT_COORD_SCALE_X,
    COURT_COORD_SCALE_Y,
    COURT_COORD_SCALE_Z,
)


def _smplh_to_coco17(joints_3d: np.ndarray, yaw: float) -> np.ndarray:
    """Convert SMPL-H joints to COCO17 with synthetic face keypoints."""
    T = joints_3d.shape[0]
    coco17 = np.zeros((T, 17, 3), dtype=np.float32)

    for coco_idx, smplh_idx in SMPLH_TO_COCO17_MAPPING.items():
        if 0 <= smplh_idx < joints_3d.shape[1]:
            coco17[:, coco_idx, :] = joints_3d[:, smplh_idx, :]

    head_idx = min(15, joints_3d.shape[1] - 1)
    head_pos = joints_3d[:, head_idx, :]
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    rot = np.array(
        [
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    for coco_idx, offset in FACE_KEYPOINT_OFFSETS.items():
        offset_arr = np.array(offset, dtype=np.float32)
        rotated_offset = offset_arr @ rot.T
        coco17[:, coco_idx, :] = head_pos + rotated_offset

    return coco17


def build_coco17_world_targets(scene: dict[str, Any]) -> np.ndarray:
    """Build world/court-coordinate COCO17 targets from a loaded scene.

    Returns:
        np.ndarray: Shape (T, 17, 3), in meters.

    """
    if "human_kp_3d" in scene:
        kp = np.asarray(scene["human_kp_3d"], dtype=np.float32)
        if kp.ndim == 2:
            kp = kp[None, ...]
        return kp

    canonical = np.asarray(scene["canonical_pose_3d"], dtype=np.float32)
    if canonical.ndim == 2:
        canonical = canonical[None, ...]

    position_norm = np.asarray(scene["position"], dtype=np.float32)
    if position_norm.ndim == 1:
        position_norm = position_norm[None, ...]
    pelvis_world = np.zeros_like(position_norm, dtype=np.float32)
    pelvis_world[:, 0] = position_norm[:, 0] * COURT_COORD_SCALE_X
    pelvis_world[:, 1] = position_norm[:, 1] * COURT_COORD_SCALE_Y
    pelvis_world[:, 2] = position_norm[:, 2] * COURT_COORD_SCALE_Z

    meta = scene.get("meta", {})
    yaw = float(meta.get("initial_yaw", 0.0))
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    rot = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    world_smplh = np.einsum("tji,ki->tjk", canonical, rot) + pelvis_world[:, None, :]
    return _smplh_to_coco17(world_smplh, yaw)
