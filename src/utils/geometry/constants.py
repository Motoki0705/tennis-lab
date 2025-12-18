"""Constants for tennis court and human pose geometry.

This module provides unified keypoint definitions used across the project:
- Court 3D keypoints (CourtKP20)
- Human keypoints (COCO-17)
- SMPL-H joint definitions
"""

from __future__ import annotations

# Import court geometry scalars for shared normalization conventions.
# NOTE: `src.utils.geometry.__init__` imports this module before `court`, but importing
# `court` here is safe because `court.py` does not import this module (no cycle).
from src.utils.geometry.court import HALF_DOUBLES_WIDTH, HALF_LENGTH, NET_HEIGHT_POST

# -----------------------------
# COCO 17 Human Keypoints (HumanKP17)
# -----------------------------

NUM_HUMAN_KP: int = 17

COCO_KP_NAMES: tuple[str, ...] = (
    "nose",  # 0
    "left_eye",  # 1
    "right_eye",  # 2
    "left_ear",  # 3
    "right_ear",  # 4
    "left_shoulder",  # 5
    "right_shoulder",  # 6
    "left_elbow",  # 7
    "right_elbow",  # 8
    "left_wrist",  # 9
    "right_wrist",  # 10
    "left_hip",  # 11
    "right_hip",  # 12
    "left_knee",  # 13
    "right_knee",  # 14
    "left_ankle",  # 15
    "right_ankle",  # 16
)

# Keypoint indices for convenience
COCO_KP_IDX: dict[str, int] = {name: i for i, name in enumerate(COCO_KP_NAMES)}

# -----------------------------
# Court 3D Keypoints (CourtKP20)
# -----------------------------

NUM_COURT_KP: int = 20

COURT_KP_NAMES: tuple[str, ...] = (
    "far_doubles_left",  # 0
    "far_doubles_right",  # 1
    "near_doubles_left",  # 2
    "near_doubles_right",  # 3
    "far_singles_left",  # 4
    "near_singles_left",  # 5
    "far_singles_right",  # 6
    "near_singles_right",  # 7
    "far_service_left",  # 8
    "far_service_right",  # 9
    "near_service_left",  # 10
    "near_service_right",  # 11
    "far_service_t",  # 12
    "near_service_t",  # 13
    "net_center",  # 14
    "left_post_base",  # 15
    "left_post_top",  # 16
    "right_post_base",  # 17
    "right_post_top",  # 18
    "center_strap_top",  # 19
)

COURT_KP_IDX: dict[str, int] = {name: i for i, name in enumerate(COURT_KP_NAMES)}

# Court skeleton connections (pairs of keypoint indices to draw lines between)
# Used for 3D court rendering with CourtKP20 indices
COURT_SKELETON: list[tuple[int, int]] = [
    # Baselines
    (0, 1),  # far doubles baseline
    (2, 3),  # near doubles baseline
    # Doubles sidelines
    (0, 2),  # left doubles sideline
    (1, 3),  # right doubles sideline
    # Singles sidelines
    (4, 5),  # left singles sideline
    (6, 7),  # right singles sideline
    # Service lines
    (8, 9),  # far service line
    (10, 11),  # near service line
    # Center service line
    (12, 13),  # center service line
    # Service line to singles corners
    (4, 8),  # far left singles to service
    (6, 9),  # far right singles to service
    (5, 10),  # near left singles to service
    (7, 11),  # near right singles to service
    # Net posts (vertical)
    (15, 16),  # left net post
    (17, 18),  # right net post
    # Net top
    (16, 19),  # left post top to center strap
    (18, 19),  # right post top to center strap
]

# -----------------------------
# Court Line Connections for 2D Rendering
# -----------------------------
# Alternative ordering used by some rendering code
# Maps to a different keypoint convention (baseline-centric ordering)

COURT_LINE_CONNECTIONS: list[tuple[int, int]] = [
    # Near baseline (near side doubles corners)
    (2, 3),
    # Near service line
    (10, 11),
    # Far service line
    (8, 9),
    # Far baseline (far side doubles corners)
    (0, 1),
    # Left doubles sideline
    (0, 2),
    # Right doubles sideline
    (1, 3),
    # Left singles sideline
    (4, 8),
    (8, 10),
    (10, 5),
    # Right singles sideline
    (6, 9),
    (9, 11),
    (11, 7),
    # Center service line (far T to near T)
    (12, 13),
    # Net line between posts
    (15, 17),
]

# -----------------------------
# SMPL-H Joint Definitions
# -----------------------------

NUM_SMPLH_BODY_JOINTS: int = 22
NUM_SMPLH_HAND_JOINTS: int = 15  # per hand
NUM_SMPLH_TOTAL_JOINTS: int = 73  # including extra joints

# SMPL-H body joint names (first 22 joints)
SMPLH_BODY_JOINT_NAMES: tuple[str, ...] = (
    "pelvis",  # 0
    "left_hip",  # 1
    "right_hip",  # 2
    "spine1",  # 3
    "left_knee",  # 4
    "right_knee",  # 5
    "spine2",  # 6
    "left_ankle",  # 7
    "right_ankle",  # 8
    "spine3",  # 9
    "left_foot",  # 10
    "right_foot",  # 11
    "neck",  # 12
    "left_collar",  # 13
    "right_collar",  # 14
    "head",  # 15
    "left_shoulder",  # 16
    "right_shoulder",  # 17
    "left_elbow",  # 18
    "right_elbow",  # 19
    "left_wrist",  # 20
    "right_wrist",  # 21
)

SMPLH_JOINT_IDX: dict[str, int] = {
    name: i for i, name in enumerate(SMPLH_BODY_JOINT_NAMES)
}

# SMPL-H to COCO 17 mapping
# For face keypoints (nose, eyes, ears), we use head joint with offsets
# Format: COCO index -> SMPL-H index (or -1 for computed from head)
SMPLH_TO_COCO17_MAPPING: dict[int, int] = {
    0: -1,  # nose -> computed from head
    1: -1,  # left_eye -> computed from head
    2: -1,  # right_eye -> computed from head
    3: -1,  # left_ear -> computed from head
    4: -1,  # right_ear -> computed from head
    5: 16,  # left_shoulder
    6: 17,  # right_shoulder
    7: 18,  # left_elbow
    8: 19,  # right_elbow
    9: 20,  # left_wrist
    10: 21,  # right_wrist
    11: 1,  # left_hip
    12: 2,  # right_hip
    13: 4,  # left_knee
    14: 5,  # right_knee
    15: 7,  # left_ankle
    16: 8,  # right_ankle
}

# Face keypoint offsets from head joint (in local head coordinate frame)
# These are approximate offsets to generate face keypoints from head center
FACE_KEYPOINT_OFFSETS: dict[int, tuple[float, float, float]] = {
    0: (0.0, 0.10, 0.0),  # nose: forward
    1: (-0.03, 0.08, 0.02),  # left_eye: slightly left, forward, up
    2: (0.03, 0.08, 0.02),  # right_eye: slightly right, forward, up
    3: (-0.07, 0.0, 0.0),  # left_ear: left
    4: (0.07, 0.0, 0.0),  # right_ear: right
}

# -----------------------------
# Court Coordinate Normalization Scales
# -----------------------------
# Shared convention for "court-coordinate normalized position" used across tasks:
#   x_norm = X / HALF_DOUBLES_WIDTH
#   y_norm = Y / HALF_LENGTH
#   z_norm = Z / NET_HEIGHT_POST
COURT_COORD_SCALE_X: float = float(HALF_DOUBLES_WIDTH)
COURT_COORD_SCALE_Y: float = float(HALF_LENGTH)
COURT_COORD_SCALE_Z: float = float(NET_HEIGHT_POST)
COURT_COORD_SCALE_XYZ: tuple[float, float, float] = (
    COURT_COORD_SCALE_X,
    COURT_COORD_SCALE_Y,
    COURT_COORD_SCALE_Z,
)
