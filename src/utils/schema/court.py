"""Unified court schema and geometry definitions.

This module consolidates all court-related definitions, including:
- ITF standard dimensions
- 3D keypoint generation
- Keypoint names and indices
- Skeleton connectivity (for rendering and visualization)
- Coordinate normalization scales
- Configurable court geometry (CourtConfig)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# -----------------------------
# ITF Standard Court Dimensions (meters)
# -----------------------------

COURT_LENGTH: float = 23.77
HALF_LENGTH: float = COURT_LENGTH / 2.0  # 11.885

SINGLES_WIDTH: float = 8.23
HALF_SINGLES_WIDTH: float = SINGLES_WIDTH / 2.0  # 4.115

DOUBLES_WIDTH: float = 10.97
HALF_DOUBLES_WIDTH: float = DOUBLES_WIDTH / 2.0  # 5.485

SERVICE_LINE_DISTANCE: float = 6.40  # Distance from net to service line
CENTER_MARK_LENGTH: float = 0.10  # Length of center mark on baseline

# Net dimensions
NET_HEIGHT_CENTER: float = 0.914  # Net height at center (3 feet)
NET_HEIGHT_POST: float = 1.07  # Net height at posts (3.5 feet)

# Net post offset from doubles sideline
NET_POST_OFFSET_X: float = 0.914


# -----------------------------
# Configurable Court Geometry
# -----------------------------

@dataclass
class CourtConfig:
    """Court geometry variation parameters.

    Allows per-scene variation of net post positions and other
    court geometry that differs across real-world venues.

    Attributes:
        net_post_offset_x: Offset of net posts from doubles sideline (m).
            ITF default is 0.914m outside. Negative values place posts
            inside the doubles sideline (common on some courts).
        net_post_offset_x_range: If set, ``net_post_offset_x`` is sampled
            uniformly from this range for each scene.
    """

    net_post_offset_x: float = NET_POST_OFFSET_X
    net_post_offset_x_range: tuple[float, float] | None = None

    def sample(self) -> CourtConfig:
        """Return a new config with stochastic parameters sampled."""
        offset = self.net_post_offset_x
        if self.net_post_offset_x_range is not None:
            lo, hi = self.net_post_offset_x_range
            offset = lo + torch.rand(1).item() * (hi - lo)
        return CourtConfig(
            net_post_offset_x=offset,
            net_post_offset_x_range=None,  # sampled config is deterministic
        )

# -----------------------------
# Fence (Run-off) Dimensions
# -----------------------------

BASELINE_CLEAR: float = 6.40
SIDELINE_CLEAR: float = 3.66
FENCE_HEIGHT: float = 3.0

X_MIN: float = -(HALF_DOUBLES_WIDTH + SIDELINE_CLEAR)  # -9.145
X_MAX: float = +(HALF_DOUBLES_WIDTH + SIDELINE_CLEAR)  # +9.145
Y_MIN: float = -(HALF_LENGTH + BASELINE_CLEAR)  # -18.285
Y_MAX: float = +(HALF_LENGTH + BASELINE_CLEAR)  # +18.285

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


def court_keypoints_3d(config: CourtConfig | None = None) -> Tensor:
    """Return 20 court keypoints (idx 0..19) as a (20, 3) tensor.

    Keypoint indices follow the CourtKP20 specification:

    0..3:  far/near doubles corners
    4..7:  far/near singles corners
    8..11: service line endpoints
    12,13: service T (far, near)
    14:    net center (ground)
    15..18: net posts (base/top, left/right)
    19:    center strap top

    Args:
        config: Optional court geometry configuration. When provided, net post
            positions use ``config.net_post_offset_x`` instead of the default.
    """
    cfg = config or CourtConfig()

    xs = HALF_SINGLES_WIDTH
    xd = HALF_DOUBLES_WIDTH
    yB = HALF_LENGTH
    yS = SERVICE_LINE_DISTANCE

    x_post_L = -(xd + cfg.net_post_offset_x)
    x_post_R = +(xd + cfg.net_post_offset_x)

    pts = [
        (-xd, +yB, 0.0),  # 0 far doubles corner left
        (+xd, +yB, 0.0),  # 1 far doubles corner right
        (-xd, -yB, 0.0),  # 2 near doubles corner left
        (+xd, -yB, 0.0),  # 3 near doubles corner right
        (-xs, +yB, 0.0),  # 4 far singles corner left
        (-xs, -yB, 0.0),  # 5 near singles corner left
        (+xs, +yB, 0.0),  # 6 far singles corner right
        (+xs, -yB, 0.0),  # 7 near singles corner right
        (-xs, +yS, 0.0),  # 8 far service-line endpoint left
        (+xs, +yS, 0.0),  # 9 far service-line endpoint right
        (-xs, -yS, 0.0),  # 10 near service-line endpoint left
        (+xs, -yS, 0.0),  # 11 near service-line endpoint right
        (0.0, +yS, 0.0),  # 12 far service T
        (0.0, -yS, 0.0),  # 13 near service T
        (0.0, 0.0, 0.0),  # 14 net center (ground)
        (x_post_L, 0.0, 0.0),  # 15 left net post base
        (x_post_L, 0.0, NET_HEIGHT_POST),  # 16 left net post top
        (x_post_R, 0.0, 0.0),  # 17 right net post base
        (x_post_R, 0.0, NET_HEIGHT_POST),  # 18 right net post top
        (0.0, 0.0, NET_HEIGHT_CENTER),  # 19 center strap top
    ]
    return torch.tensor(pts, dtype=torch.float32)


# -----------------------------
# Court Skeleton / Connectivity
# -----------------------------

# Unified dictionary of court lines defined as pairs of keypoint indices.
# This covers all standard court lines plus net structure.
COURT_SKELETON: list[tuple[int, int]] = [
    # --- Baselines ---
    (0, 1),  # far doubles baseline
    (2, 3),  # near doubles baseline
    
    # --- Doubles Sidelines ---
    (0, 2),  # left doubles sideline
    (1, 3),  # right doubles sideline
    
    # --- Singles Sidelines ---
    # Split into 3 segments: (far baseline -> far service), (far service -> near service), (near service -> near baseline)
    # Using existing keypoints for better connectivity graph
    # Far part
    (4, 8),  # far left singles corner -> far left service
    (6, 9),  # far right singles corner -> far right service
    # Middle part (Service box sides)
    (8, 10), # far left service -> near left service
    (9, 11), # far right service -> near right service
    # Near part
    (10, 5), # near left service -> near left singles corner
    (11, 7), # near right service -> near right singles corner
    
    # --- Service Lines ---
    (8, 9),   # far service line
    (10, 11), # near service line
    
    # --- Center Service Line ---
    (12, 13), # far T -> near T
    
    # --- Net Structure ---
    # Net posts (vertical)
    (15, 16), # left net post
    (17, 18), # right net post
    # Net top cable (via center strap)
    (16, 19), # left post top -> center strap
    (19, 18), # center strap -> right post top
]




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
