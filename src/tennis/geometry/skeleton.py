"""Skeleton specification for tennis pose: ViTPose-COCO 17 + racket 3 points."""

from __future__ import annotations

VITPOSE_17_NAMES: list[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

RACKET_3_NAMES: list[str] = [
    "racket_handle",
    "racket_throat",
    "racket_head_top",
]


def joint_names() -> list[str]:
    """Return a copy of the ViTPose 17-keypoint name list."""
    return list(VITPOSE_17_NAMES)


def racket_names() -> list[str]:
    """Return a copy of the 3 racket keypoint names."""
    return list(RACKET_3_NAMES)


def all_keypoint_names() -> list[str]:
    """Return body + racket keypoint names in canonical order."""
    return joint_names() + racket_names()


def name_to_index() -> dict[str, int]:
    """Return a mapping from keypoint name to its global index."""
    return {name: i for i, name in enumerate(all_keypoint_names())}


# COCO-style skeleton bone pairs (indices within the 17-joint set)
COCO_BONES: list[tuple[int, int]] = [
    (5, 7),  # left_shoulder - left_elbow
    (7, 9),  # left_elbow - left_wrist
    (6, 8),  # right_shoulder - right_elbow
    (8, 10),  # right_elbow - right_wrist
    (5, 6),  # left_shoulder - right_shoulder
    (5, 11),  # left_shoulder - left_hip
    (6, 12),  # right_shoulder - right_hip
    (11, 12),  # left_hip - right_hip
    (11, 13),  # left_hip - left_knee
    (13, 15),  # left_knee - left_ankle
    (12, 14),  # right_hip - right_knee
    (14, 16),  # right_knee - right_ankle
    (0, 5),  # nose - left_shoulder (approx chest connection via neck)
    (0, 6),  # nose - right_shoulder
]


# Indices for right-handed assumption; left-handed can swap wrists if needed
RIGHT_WRIST_IDX = VITPOSE_17_NAMES.index("right_wrist")
LEFT_WRIST_IDX = VITPOSE_17_NAMES.index("left_wrist")
RACKET_HANDLE_IDX = 17
