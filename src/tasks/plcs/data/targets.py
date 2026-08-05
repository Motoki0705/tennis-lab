"""Target-generation utilities for PLCS datasets."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np

from src.utils.schema.court import (
    COURT_COORD_SCALE_X,
    COURT_COORD_SCALE_Y,
    COURT_COORD_SCALE_Z,
)
from src.utils.schema.player import FACE_KEYPOINT_OFFSETS, SMPLH_TO_COCO17_MAPPING


def _smplh_to_coco17(joints_3d: np.ndarray, yaw: float | np.ndarray) -> np.ndarray:
    """Convert SMPL-H joints to COCO17 with synthetic face keypoints."""
    T = joints_3d.shape[0]
    coco17 = np.zeros((T, 17, 3), dtype=np.float32)

    for coco_idx, smplh_idx in SMPLH_TO_COCO17_MAPPING.items():
        if 0 <= smplh_idx < joints_3d.shape[1]:
            coco17[:, coco_idx, :] = joints_3d[:, smplh_idx, :]

    head_idx = min(15, joints_3d.shape[1] - 1)
    head_pos = joints_3d[:, head_idx, :]
    yaw_arr = np.asarray(yaw, dtype=np.float32)
    if yaw_arr.ndim == 0:
        yaw_arr = np.full((T,), float(yaw_arr), dtype=np.float32)
    elif yaw_arr.shape != (T,):
        raise ValueError(f"yaw must be scalar or shape ({T},), got {yaw_arr.shape}")
    cos_yaw = np.cos(yaw_arr).astype(np.float32)
    sin_yaw = np.sin(yaw_arr).astype(np.float32)

    for coco_idx, offset in FACE_KEYPOINT_OFFSETS.items():
        offset_arr = np.array(offset, dtype=np.float32)
        rotated_offset = np.stack(
            [
                offset_arr[0] * cos_yaw - offset_arr[1] * sin_yaw,
                offset_arr[0] * sin_yaw + offset_arr[1] * cos_yaw,
                np.full((T,), offset_arr[2], dtype=np.float32),
            ],
            axis=1,
        )
        coco17[:, coco_idx, :] = head_pos + rotated_offset

    return cast(np.ndarray, coco17)


def build_coco17_world_targets(scene: dict[str, Any]) -> np.ndarray:
    """Build world/court-coordinate COCO17 targets from a loaded scene.

    Returns:
        np.ndarray: Shape (T, 17, 3), in meters.

    """
    if "human_kp_3d" in scene:
        kp = np.asarray(scene["human_kp_3d"], dtype=np.float32)
        if kp.ndim == 2:
            kp = kp[None, ...]
        return cast(np.ndarray, kp)

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
    init_yaw = float(meta.get("initial_yaw", 0.0))

    # Use per-frame rotation (cos, sin) when available.
    if "rotation" in scene:
        rot_cs = np.asarray(scene["rotation"], dtype=np.float32)
        if rot_cs.ndim == 1 and rot_cs.shape[0] == 2:
            cos_yaw = np.full((canonical.shape[0],), rot_cs[0], dtype=np.float32)
            sin_yaw = np.full((canonical.shape[0],), rot_cs[1], dtype=np.float32)
        elif rot_cs.ndim == 2 and rot_cs.shape[1] == 2:
            cos_yaw = rot_cs[:, 0].astype(np.float32)
            sin_yaw = rot_cs[:, 1].astype(np.float32)
        else:
            cos_yaw = np.full(
                (canonical.shape[0],), math.cos(init_yaw), dtype=np.float32
            )
            sin_yaw = np.full(
                (canonical.shape[0],), math.sin(init_yaw), dtype=np.float32
            )
    else:
        cos_yaw = np.full((canonical.shape[0],), math.cos(init_yaw), dtype=np.float32)
        sin_yaw = np.full((canonical.shape[0],), math.sin(init_yaw), dtype=np.float32)

    world_smplh = np.empty_like(canonical, dtype=np.float32)
    world_smplh[..., 0] = (
        canonical[..., 0] * cos_yaw[:, None]
        - canonical[..., 1] * sin_yaw[:, None]
        + pelvis_world[:, None, 0]
    )
    world_smplh[..., 1] = (
        canonical[..., 0] * sin_yaw[:, None]
        + canonical[..., 1] * cos_yaw[:, None]
        + pelvis_world[:, None, 1]
    )
    world_smplh[..., 2] = canonical[..., 2] + pelvis_world[:, None, 2]

    yaw_for_face = np.arctan2(sin_yaw, cos_yaw).astype(np.float32)
    return _smplh_to_coco17(world_smplh, yaw_for_face)
