"""Canonical public API for BLCS model I/O composition and contracts."""

from src.tasks.blcs.model_io.adapters import (
    AxialTrajectoryModelIOAdapter,
    MultiViewTrajectoryModelIOAdapter,
    SingleTrajectoryModelIOAdapter,
    TrackQueryAblationModelIOAdapter,
    TrackQueryModelIOAdapter,
    TrackQueryReferenceAblationModelIOAdapter,
    TrackQueryReferenceModelIOAdapter,
    TrajectoryModelIOAdapter,
)
from src.tasks.blcs.model_io.checkpoints import (
    resolve_blcs_track_query_reference_contract,
    validate_blcs_checkpoint_track_query_reference,
    write_blcs_checkpoint_track_query_reference,
)
from src.tasks.blcs.model_io.contracts import (
    BLCSReferenceMetadata,
    BLCSTrackQueryPrediction,
    BLCSTrackQueryTrainingBatch,
    BLCSTrajectoryPrediction,
    BLCSTrajectoryTrainingBatch,
    blcs_reference_metadata_from_batch,
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
    "BLCSReferenceMetadata",
    "BLCSTrackQueryPrediction",
    "BLCSTrackQueryTrainingBatch",
    "BLCSTrajectoryPrediction",
    "BLCSTrajectoryTrainingBatch",
    "blcs_reference_metadata_from_batch",
    "blcs_track_query_prediction_to_physical",
    "blcs_trajectory_prediction_to_physical",
    "MultiViewTrajectoryModelIOAdapter",
    "SingleTrajectoryModelIOAdapter",
    "TrackQueryAblationModelIOAdapter",
    "TrackQueryBoundModelIO",
    "TrackQueryModelIOAdapter",
    "TrackQueryReferenceAblationModelIOAdapter",
    "TrackQueryReferenceModelIOAdapter",
    "TrajectoryBoundModelIO",
    "TrajectoryModelIOAdapter",
    "compose_blcs_model_io",
    "compose_blcs_track_query_model_io",
    "compose_blcs_trajectory_model_io",
    "resolve_blcs_track_query_reference_contract",
    "validate_blcs_checkpoint_track_query_reference",
    "write_blcs_checkpoint_track_query_reference",
]
