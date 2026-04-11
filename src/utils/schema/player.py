"""Shared human pose schema definitions.
 
This module provides unified schema definitions for human pose:
- Human keypoints (COCO-17)
- SMPL-H joint definitions
"""
 
from __future__ import annotations

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

# COCO 17-keypoint skeleton connections
COCO17_SKELETON: list[tuple[int, int]] = [
    # Head
    (0, 1),  # nose -> left_eye
    (0, 2),  # nose -> right_eye
    (1, 3),  # left_eye -> left_ear
    (2, 4),  # right_eye -> right_ear
    # Torso
    (5, 6),  # left_shoulder -> right_shoulder
    (5, 11),  # left_shoulder -> left_hip
    (6, 12),  # right_shoulder -> right_hip
    (11, 12),  # left_hip -> right_hip
    # Left arm
    (5, 7),  # left_shoulder -> left_elbow
    (7, 9),  # left_elbow -> left_wrist
    # Right arm
    (6, 8),  # right_shoulder -> right_elbow
    (8, 10),  # right_elbow -> right_wrist
    # Left leg
    (11, 13),  # left_hip -> left_knee
    (13, 15),  # left_knee -> left_ankle
    # Right leg
    (12, 14),  # right_hip -> right_knee
    (14, 16),  # right_knee -> right_ankle
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

# SMPL-H 22-joint skeleton connections (body joints only, excluding hands)
SMPLH_SKELETON: list[tuple[int, int]] = [
    # Spine
    (0, 3),  # pelvis -> spine1
    (3, 6),  # spine1 -> spine2
    (6, 9),  # spine2 -> spine3
    (9, 12),  # spine3 -> neck
    (12, 15),  # neck -> head
    # Left side
    (0, 1),  # pelvis -> left_hip
    (1, 4),  # left_hip -> left_knee
    (4, 7),  # left_knee -> left_ankle
    (7, 10),  # left_ankle -> left_foot
    (9, 13),  # spine3 -> left_collar
    (13, 16),  # left_collar -> left_shoulder
    (16, 18),  # left_shoulder -> left_elbow
    (18, 20),  # left_elbow -> left_wrist
    # Right side
    (0, 2),  # pelvis -> right_hip
    (2, 5),  # right_hip -> right_knee
    (5, 8),  # right_knee -> right_ankle
    (8, 11),  # right_ankle -> right_foot
    (9, 14),  # spine3 -> right_collar
    (14, 17),  # right_collar -> right_shoulder
    (17, 19),  # right_shoulder -> right_elbow
    (19, 21),  # right_elbow -> right_wrist
]

# SMPL 24-joint skeleton (standard SMPL body model)
SMPL_SKELETON: list[tuple[int, int]] = [
    # Spine
    (0, 3),  # pelvis -> spine1
    (3, 6),  # spine1 -> spine2
    (6, 9),  # spine2 -> spine3
    (9, 12),  # spine3 -> neck
    (12, 15),  # neck -> head
    # Left side
    (0, 1),  # pelvis -> left_hip
    (1, 4),  # left_hip -> left_knee
    (4, 7),  # left_knee -> left_ankle
    (7, 10),  # left_ankle -> left_foot
    (9, 13),  # spine3 -> left_collar
    (13, 16),  # left_collar -> left_shoulder
    (16, 18),  # left_shoulder -> left_elbow
    (18, 20),  # left_elbow -> left_wrist
    (20, 22),  # left_wrist -> left_hand
    # Right side
    (0, 2),  # pelvis -> right_hip
    (2, 5),  # right_hip -> right_knee
    (5, 8),  # right_knee -> right_ankle
    (8, 11),  # right_ankle -> right_foot
    (9, 14),  # spine3 -> right_collar
    (14, 17),  # right_collar -> right_shoulder
    (17, 19),  # right_shoulder -> right_elbow
    (19, 21),  # right_elbow -> right_wrist
    (21, 23),  # right_wrist -> right_hand
]

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

# Human3.6M 17-joint to COCO 17 mapping
# H3.6M: 0=pelvis, 1=R_hip, 2=R_knee, 3=R_ankle, 4=L_hip, 5=L_knee,
#         6=L_ankle, 7=spine, 8=neck, 9=head, 10=head_top, 11=L_shoulder,
#         12=L_elbow, 13=L_wrist, 14=R_shoulder, 15=R_elbow, 16=R_wrist
# COCO index -> H3.6M index (or -1 for computed from head)
H36M_TO_COCO17_MAPPING: dict[int, int] = {
    0: -1,  # nose -> computed from head
    1: -1,  # left_eye -> computed from head
    2: -1,  # right_eye -> computed from head
    3: -1,  # left_ear -> computed from head
    4: -1,  # right_ear -> computed from head
    5: 11,  # left_shoulder
    6: 14,  # right_shoulder
    7: 12,  # left_elbow
    8: 15,  # right_elbow
    9: 13,  # left_wrist
    10: 16,  # right_wrist
    11: 4,  # left_hip
    12: 1,  # right_hip
    13: 5,  # left_knee
    14: 2,  # right_knee
    15: 6,  # left_ankle
    16: 3,  # right_ankle
}

# H3.6M joint index for the head (used for face keypoint synthesis)
H36M_HEAD_JOINT: int = 9
H36M_HEAD_TOP_JOINT: int = 10

# Face keypoint offsets from head joint (in local head coordinate frame)
# These are approximate offsets to generate face keypoints from head center
FACE_KEYPOINT_OFFSETS: dict[int, tuple[float, float, float]] = {
    0: (0.0, 0.10, 0.0),  # nose: forward
    1: (-0.03, 0.08, 0.02),  # left_eye: slightly left, forward, up
    2: (0.03, 0.08, 0.02),  # right_eye: slightly right, forward, up
    3: (-0.07, 0.0, 0.0),  # left_ear: left
    4: (0.07, 0.0, 0.0),  # right_ear: right
}
