"""BLCS simulation module.

Provides physics simulation and shot generation for ball trajectory data.
"""

from src.tasks.blcs.generate_dataset.simulation.ball_physics import (
    BallPhysics,
    BallState,
    PhysicsConfig,
)
from src.tasks.blcs.generate_dataset.simulation.cell_manager import (
    NUM_CELLS_PER_SIDE,
    NUM_IN_COURT_CELLS,
    NUM_OUT_COURT_CELLS,
    NUM_TOTAL_CELLS,
    CellManager,
    ShotCategory,
)
from src.tasks.blcs.generate_dataset.simulation.rally_simulator import (
    RallyConfig,
    RallyEndReason,
    RallyResult,
    RallySimulator,
    ShotEventInfo,
    ShotResult,
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
    "ShotResult",
    "ShotType",
    "RallySimulator",
    "RallyConfig",
    "RallyResult",
    "RallyEndReason",
    "ShotEventInfo",
]
