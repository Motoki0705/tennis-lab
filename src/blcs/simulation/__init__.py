"""BLCS simulation module.

Provides physics simulation and shot generation for ball trajectory data.
"""

from src.blcs.simulation.ball_physics import BallPhysics, BallState, PhysicsConfig
from src.blcs.simulation.cell_manager import CellManager, ShotCategory
from src.blcs.simulation.shot_simulator import ShotConfig, ShotResult, ShotSimulator

__all__ = [
    "BallPhysics",
    "BallState",
    "PhysicsConfig",
    "CellManager",
    "ShotCategory",
    "ShotSimulator",
    "ShotConfig",
    "ShotResult",
]
