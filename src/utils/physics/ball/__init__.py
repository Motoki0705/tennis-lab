"""Shared tennis-ball physics: force model, bounce and free-flight integration."""

from src.utils.physics.ball.bounce import ground_bounce
from src.utils.physics.ball.dynamics import (
    BallField,
    acceleration,
    semi_implicit_euler_step,
)
from src.utils.physics.ball.integrate import integrate_flight
from src.utils.physics.ball.surfaces import (
    SURFACE_NAMES,
    SURFACES,
    TENNIS_BALL,
    BallProperties,
    SurfaceProperties,
    surface_properties,
)

__all__ = [
    "SURFACES",
    "SURFACE_NAMES",
    "TENNIS_BALL",
    "BallField",
    "BallProperties",
    "SurfaceProperties",
    "acceleration",
    "ground_bounce",
    "integrate_flight",
    "semi_implicit_euler_step",
    "surface_properties",
]
