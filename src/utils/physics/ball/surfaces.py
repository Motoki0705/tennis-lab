"""Fixed tennis-ball and court-surface constants (the single source of truth).

Ball (Cross 2002, Table I): radius 33 mm and moment of inertia
``I = 0.55 m R^2`` for a hollow ball with a 6 mm wall.  The normal reaction acts
``D = 4 mm`` ahead of the centre of mass, "typically about 4 mm for a low speed
bounce but it can be as large as 11 mm" on clay (Cross 2003); the bounce model
uses the typical value on every surface.

Surfaces: ``restitution`` is the vertical coefficient of restitution
``e_y = -v_z2 / v_z1``, ``friction`` the coefficient of sliding friction and
``grip_restitution`` the horizontal coefficient of restitution ``e_x`` of the
contact point when the ball grips the court.

- clay: COF typically about 0.8 (Cross 2003) and COR about 0.9 for oblique
  impacts (Cross 2003; Cross 2010 measured 0.91).
- grass: COF 0.53 measured at 30 m/s, 17 deg (Cross 2010, Fig. 3) and COR about
  0.75 (Cross 2010, Sec. 7).
- hard: ITF Category 3 "medium" (CPR 35-39) at the average COR 0.81 that zeroes
  the ITF perception term; ``CPR = 100 (1 - mu) + 150 (0.81 - COR)`` gives
  CPR 37 for COF 0.63.  The two other surfaces fall in ITF Category 1 (clay,
  CPR 6.5) and Category 5 (grass, CPR 56).
- ``e_x``: 0.10-0.17 for gripping tennis-ball bounces (Cross 2002, Table III);
  no surface-resolved values are published, so all surfaces share 0.15.

References:
    R. Cross, "Grip-slip behavior of a bouncing ball", Am. J. Phys. 70, 1093 (2002).
    R. Cross, "Measurements of the horizontal and vertical speeds of tennis
    courts", Sports Engineering 6, 93-109 (2003).
    R. Cross, "Measurement of the speed and bounce of tennis courts" (2010),
    https://www.physics.usyd.edu.au/~cross/PUBLICATIONS/52.%20SpeedAndBounce.pdf
    ITF Court Pace Classification Programme categories 1-5.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class BallProperties:
    radius_m: float
    inertia_factor: float
    normal_force_offset_m: float


@dataclass(frozen=True)
class SurfaceProperties:
    restitution: float
    friction: float
    grip_restitution: float

    def __post_init__(self) -> None:
        if not 0 < self.restitution <= 1 or self.friction < 0:
            raise ValueError("Require restitution in (0,1] and friction >= 0")
        if not -1 <= self.grip_restitution <= 1:
            raise ValueError("grip_restitution must lie in [-1, 1]")


TENNIS_BALL = BallProperties(
    radius_m=0.033, inertia_factor=0.55, normal_force_offset_m=0.004
)

SURFACES = MappingProxyType(
    {
        "hard": SurfaceProperties(
            restitution=0.81, friction=0.63, grip_restitution=0.15
        ),
        "clay": SurfaceProperties(
            restitution=0.90, friction=0.80, grip_restitution=0.15
        ),
        "grass": SurfaceProperties(
            restitution=0.75, friction=0.53, grip_restitution=0.15
        ),
    }
)
# Fixed class order for models that classify the surface.
SURFACE_NAMES: tuple[str, ...] = ("hard", "clay", "grass")


def surface_properties(name: str) -> SurfaceProperties:
    if name not in SURFACES:
        raise KeyError(f"Unknown surface {name!r}; expected one of {SURFACE_NAMES}")
    return SURFACES[name]
