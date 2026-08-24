"""Canonical public API for BLCS model I/O composition and contracts."""

from src.tasks.blcs.model_io.adapters import (
    AxialTrajectoryModelIOAdapter,
    MultiViewTrajectoryModelIOAdapter,
    SingleTrajectoryModelIOAdapter,
    TrackQueryAblationModelIOAdapter,
    TrackQueryModelIOAdapter,
    TrajectoryModelIOAdapter,
)
from src.tasks.blcs.model_io.contracts import (
    BLCSTrackQueryPrediction,
    BLCSTrackQueryTrainingBatch,
    BLCSTrajectoryPrediction,
    BLCSTrajectoryTrainingBatch,
    blcs_track_query_prediction_to_physical,
    blcs_trajectory_prediction_to_physical,
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
    "blcs_track_query_prediction_to_physical",
    "blcs_trajectory_prediction_to_physical",
    "MultiViewTrajectoryModelIOAdapter",
    "SingleTrajectoryModelIOAdapter",
    "TrackQueryAblationModelIOAdapter",
    "TrackQueryBoundModelIO",
    "TrackQueryModelIOAdapter",
    "TrajectoryBoundModelIO",
    "TrajectoryModelIOAdapter",
    "compose_blcs_model_io",
    "compose_blcs_track_query_model_io",
    "compose_blcs_trajectory_model_io",
]
