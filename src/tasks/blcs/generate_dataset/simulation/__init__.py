"""BLCS simulation module.

Provides physics simulation and shot generation for ball trajectory data.
"""

from src.tasks.blcs.generate_dataset.ball_physics import BallPhysics, BallState, PhysicsConfig
from src.tasks.blcs.generate_dataset.cell_manager import (
    CellManager,
    NUM_CELLS_PER_SIDE,
    NUM_IN_COURT_CELLS,
    NUM_OUT_COURT_CELLS,
    NUM_TOTAL_CELLS,
    ShotCategory,
)
from src.tasks.blcs.generate_dataset.rally_simulator import (
    RallyConfig,
    RallyEndReason,
    RallyResult,
    RallySimulator,
    ShotEventInfo,
)
from src.tasks.blcs.generate_dataset.shot_simulator import (
    ShotConfig,
    ShotResult,
    ShotSimulator,
    ShotType,
)

__all__ = [
    "BallPhysics",
    "BallState",
    "PhysicsConfig",
    "CellManager",
    "NUM_CELLS_PER_SIDE",
    "NUM_IN_COURT_CELLS",
    "NUM_OUT_COURT_CELLS",
    "NUM_TOTAL_CELLS",
    "ShotCategory",
    "ShotSimulator",
    "ShotConfig",
    "ShotResult",
    "ShotType",
    "RallySimulator",
    "RallyConfig",
    "RallyResult",
    "RallyEndReason",
    "ShotEventInfo",
]
