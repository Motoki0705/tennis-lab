"""Constants for PLCS module.

This module re-exports from src.utils.geometry.constants for backward compatibility.

Note:
    New code should import from src.utils.geometry directly.

"""

# Re-export all keypoint and geometry constants
from src.utils.geometry.constants import (
    # Human keypoints (COCO-17)
    COCO_KP_IDX,
    COCO_KP_NAMES,
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

__all__ = [
    # Court keypoints
    "NUM_COURT_KP",
    "COURT_KP_NAMES",
    "COURT_KP_IDX",
    "COURT_SKELETON",
    "COURT_LINE_CONNECTIONS",
    # Human keypoints
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
]
