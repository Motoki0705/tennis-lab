"""Court geometry helpers shared by the tennis simulator.

Standard tennis court dimensions according to ITF regulations.
All measurements are in meters.

Court coordinate system:
- Origin at center of court (net center)
- X-axis: sideline direction (positive = right when facing net)
- Y-axis: baseline direction (positive = far side)
- Z-axis: vertical (positive = up)
"""

from __future__ import annotations

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
# Fence (Run-off) Dimensions
# -----------------------------

BASELINE_CLEAR: float = 6.40
SIDELINE_CLEAR: float = 3.66
FENCE_HEIGHT: float = 3.0

X_MIN: float = -(HALF_DOUBLES_WIDTH + SIDELINE_CLEAR)  # -9.145
X_MAX: float = +(HALF_DOUBLES_WIDTH + SIDELINE_CLEAR)  # +9.145
Y_MIN: float = -(HALF_LENGTH + BASELINE_CLEAR)  # -18.285
Y_MAX: float = +(HALF_LENGTH + BASELINE_CLEAR)  # +18.285


def court_keypoints_3d() -> Tensor:
    """Return 20 court keypoints (idx 0..19) as a (20, 3) tensor.

    Keypoint indices follow the CourtKP20 specification:

    0..3:  far/near doubles corners
    4..7:  far/near singles corners
    8..11: service line endpoints
    12,13: service T (far, near)
    14:    net center (ground)
    15..18: net posts (base/top, left/right)
    19:    center strap top
    """
    xs = HALF_SINGLES_WIDTH
    xd = HALF_DOUBLES_WIDTH
    yB = HALF_LENGTH
    yS = SERVICE_LINE_DISTANCE

    x_post_L = -(xd + NET_POST_OFFSET_X)
    x_post_R = +(xd + NET_POST_OFFSET_X)

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

