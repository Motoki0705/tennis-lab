"""Canonical public API for BLCS model I/O composition and contracts."""

from src.tasks.blcs.model_io.adapters import (
    AxialTrajectoryModelIOAdapter,
    MultiViewTrajectoryModelIOAdapter,
    SingleTrajectoryModelIOAdapter,
    TrackQueryModelIOAdapter,
    TrajectoryModelIOAdapter,
)
from src.tasks.blcs.model_io.contracts import (
    BLCSTrackQueryPrediction,
    BLCSTrackQueryTrainingBatch,
    BLCSTrajectoryPrediction,
    BLCSTrajectoryTrainingBatch,
)
from src.tasks.blcs.model_io.factory import (
    BLCSBoundModelIO,
    TrackQueryBoundModelIO,
    TrajectoryBoundModelIO,
    compose_blcs_model_io,
    compose_blcs_track_query_model_io,
    compose_blcs_trajectory_model_io,
)

__all__ = [
    "AxialTrajectoryModelIOAdapter",
    "BLCSBoundModelIO",
    "BLCSTrackQueryPrediction",
    "BLCSTrackQueryTrainingBatch",
    "BLCSTrajectoryPrediction",
    "BLCSTrajectoryTrainingBatch",
    "MultiViewTrajectoryModelIOAdapter",
    "SingleTrajectoryModelIOAdapter",
    "TrackQueryBoundModelIO",
    "TrackQueryModelIOAdapter",
    "TrajectoryBoundModelIO",
    "TrajectoryModelIOAdapter",
    "compose_blcs_model_io",
    "compose_blcs_track_query_model_io",
    "compose_blcs_trajectory_model_io",
]
