"""Canonical public API for BLCS model I/O composition and contracts."""

from src.tasks.blcs.model_io.adapters import (
    AxialTrajectoryModelIOAdapter,
    TrajectoryModelIOAdapter,
)
from src.tasks.blcs.model_io.contracts import (
    BLCSReferenceMetadata,
    BLCSTrajectoryPrediction,
    BLCSTrajectoryTrainingBatch,
    blcs_reference_metadata_from_batch,
    blcs_trajectory_prediction_to_physical,
)
from src.tasks.blcs.model_io.factory import (
    BLCSBoundModelIO,
    TrajectoryBoundModelIO,
    compose_blcs_model_io,
    compose_blcs_trajectory_model_io,
)

__all__ = [
    "AxialTrajectoryModelIOAdapter",
    "BLCSBoundModelIO",
    "BLCSReferenceMetadata",
    "BLCSTrajectoryPrediction",
    "BLCSTrajectoryTrainingBatch",
    "blcs_reference_metadata_from_batch",
    "blcs_trajectory_prediction_to_physical",
    "TrajectoryBoundModelIO",
    "TrajectoryModelIOAdapter",
    "compose_blcs_model_io",
    "compose_blcs_trajectory_model_io",
]
