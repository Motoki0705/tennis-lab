"""Geometry utilities for tennis court and human pose.

This module provides:
- Court dimensions and keypoint definitions (ITF standard)
- Human keypoint definitions (COCO-17, SMPL-H)
"""

from src.utils.schema.keypoint_schema import (
    # Human keypoints (COCO-17)
    COCO_KP_IDX,
    COCO_KP_NAMES,
    # Court coordinate normalization scales
    COURT_COORD_SCALE_X,
    COURT_COORD_SCALE_XYZ,
    COURT_COORD_SCALE_Y,
    COURT_COORD_SCALE_Z,
    # Court keypoints
    COURT_KP_IDX,
    COURT_KP_NAMES,
    COURT_LINE_CONNECTIONS,
    COURT_SKELETON,
    # SMPL-H
    FACE_KEYPOINT_OFFSETS,
    NUM_COURT_KP,
    NUM_HUMAN_KP,
    NUM_SMPLH_BODY_JOINTS,
    NUM_SMPLH_HAND_JOINTS,
    NUM_SMPLH_TOTAL_JOINTS,
    SMPLH_BODY_JOINT_NAMES,
    SMPLH_JOINT_IDX,
    SMPLH_TO_COCO17_MAPPING,
)
from src.utils.geometry.court import (
    # Court dimensions
    BASELINE_CLEAR,
    CENTER_MARK_LENGTH,
    COURT_LENGTH,
    DOUBLES_WIDTH,
    FENCE_HEIGHT,
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    HALF_SINGLES_WIDTH,
    NET_HEIGHT_CENTER,
    NET_HEIGHT_POST,
    NET_POST_OFFSET_X,
    SERVICE_LINE_DISTANCE,
    SIDELINE_CLEAR,
    SINGLES_WIDTH,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    court_keypoints_3d,
)

__all__ = [
    # Court dimensions
    "COURT_LENGTH",
    "HALF_LENGTH",
    "SINGLES_WIDTH",
    "HALF_SINGLES_WIDTH",
    "DOUBLES_WIDTH",
    "HALF_DOUBLES_WIDTH",
    "SERVICE_LINE_DISTANCE",
    "CENTER_MARK_LENGTH",
    "NET_HEIGHT_CENTER",
    "NET_HEIGHT_POST",
    "NET_POST_OFFSET_X",
    # Fence dimensions
    "BASELINE_CLEAR",
    "SIDELINE_CLEAR",
    "FENCE_HEIGHT",
    "X_MIN",
    "X_MAX",
    "Y_MIN",
    "Y_MAX",
    # Court keypoints
    "NUM_COURT_KP",
    "COURT_KP_NAMES",
    "COURT_KP_IDX",
    "COURT_SKELETON",
    "COURT_LINE_CONNECTIONS",
    "court_keypoints_3d",
    # Human keypoints (COCO-17)
    "NUM_HUMAN_KP",
    "COCO_KP_NAMES",
    "COCO_KP_IDX",
    # SMPL-H
    "NUM_SMPLH_BODY_JOINTS",
    "NUM_SMPLH_HAND_JOINTS",
    "NUM_SMPLH_TOTAL_JOINTS",
    "SMPLH_BODY_JOINT_NAMES",
    "SMPLH_JOINT_IDX",
    "SMPLH_TO_COCO17_MAPPING",
    "FACE_KEYPOINT_OFFSETS",
    # Court coordinate normalization scales
    "COURT_COORD_SCALE_X",
    "COURT_COORD_SCALE_Y",
    "COURT_COORD_SCALE_Z",
    "COURT_COORD_SCALE_XYZ",
]
