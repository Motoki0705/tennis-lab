"""Shared loss primitives reused across tasks.

Submodules:

- :mod:`temporal` — masked finite-difference smoothness / ballistic priors on
  per-frame coordinate sequences (used by BLCS ball and PLCS player position).
"""

from src.utils.losses.temporal import (
    BallisticGravityPenalty,
    TemporalSmoothnessPenalty,
    ballistic_second_difference,
    finite_difference,
)

__all__ = [
    "BallisticGravityPenalty",
    "TemporalSmoothnessPenalty",
    "ballistic_second_difference",
    "finite_difference",
]
