"""Constants for BLCS module.

Ball physics and normalization constants based on the BLCS specification.

Note:
    For court geometry constants (HALF_DOUBLES_WIDTH, HALF_LENGTH, etc.),
    import directly from src.utils.geometry.
"""

from __future__ import annotations

from src.utils.geometry import HALF_DOUBLES_WIDTH, HALF_LENGTH, NET_HEIGHT_POST

# -----------------------------
# Ball Physical Constants
# -----------------------------

BALL_DIAMETER = 0.067  # m (tennis ball diameter: 6.54-6.86 cm)
BALL_RADIUS = BALL_DIAMETER / 2.0
BALL_MASS = 0.058  # kg (tennis ball mass: 56.0-59.4 g)

# Air drag coefficient (approximate for tennis ball)
DRAG_COEFFICIENT = 0.5
AIR_DENSITY = 1.2  # kg/m^3

# Gravity
GRAVITY = 9.81  # m/s^2

# Coefficient of restitution for bounce
COR_COURT = 0.75  # tennis ball on hard court

# -----------------------------
# Sequence Constants
# -----------------------------

MAX_SEQ_LEN = 120  # Maximum sequence length (4 seconds at 30 fps)
MIN_SEQ_LEN = 15  # Minimum sequence length (0.5 seconds at 30 fps)
DEFAULT_FPS = 30  # Default frame rate

# -----------------------------
# Normalization Scales
# -----------------------------
# From docs/blcs.md:
# x = X / HALF_DOUBLES_WIDTH
# y = Y / HALF_LENGTH
# z = Z / NET_HEIGHT_POST

NORM_SCALE_X = HALF_DOUBLES_WIDTH  # 5.485 m
NORM_SCALE_Y = HALF_LENGTH  # 11.885 m
NORM_SCALE_Z = NET_HEIGHT_POST  # 1.07 m

# -----------------------------
# Ball trajectory bounds (in normalized coordinates)
# -----------------------------

# Ball can go outside court but not too far
BALL_X_MIN = -2.0  # ~11m outside court
BALL_X_MAX = 2.0
BALL_Y_MIN = -2.0  # ~24m outside court
BALL_Y_MAX = 2.0
BALL_Z_MIN = 0.0  # ground level
BALL_Z_MAX = 10.0  # ~10.7m high (lob shots)

__all__ = [
    # Ball constants
    "BALL_DIAMETER",
    "BALL_RADIUS",
    "BALL_MASS",
    "DRAG_COEFFICIENT",
    "AIR_DENSITY",
    "GRAVITY",
    "COR_COURT",
    # Sequence constants
    "MAX_SEQ_LEN",
    "MIN_SEQ_LEN",
    "DEFAULT_FPS",
    # Normalization
    "NORM_SCALE_X",
    "NORM_SCALE_Y",
    "NORM_SCALE_Z",
    # Bounds
    "BALL_X_MIN",
    "BALL_X_MAX",
    "BALL_Y_MIN",
    "BALL_Y_MAX",
    "BALL_Z_MIN",
    "BALL_Z_MAX",
]
