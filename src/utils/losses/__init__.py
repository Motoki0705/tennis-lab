"""Shared loss primitives reused across tasks.

Submodules:

- :mod:`temporal` — masked finite-difference smoothness / ballistic priors on
  per-frame coordinate sequences (used by BLCS ball and PLCS player position).
"""

from src.utils.losses.temporal import (
    ballistic_gravity_penalty,
    ballistic_second_difference,
    finite_difference,
    smoothness_penalty,
)

__all__ = [
    "ballistic_gravity_penalty",
    "ballistic_second_difference",
    "finite_difference",
    "smoothness_penalty",
]
