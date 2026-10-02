"""Canonical public PLCS model I/O composition API."""

from src.tasks.plcs.model_io.adapters import PLCSAdapter, PLCSModelIOAdapter
from src.tasks.plcs.model_io.contracts import (
    PLCSDecodedPrediction,
    PLCSInputProfile,
    PLCSPhysicalPrediction,
    PLCSPreparedBatch,
    PLCSReferenceMetadata,
    PLCSReprojectionTarget,
    plcs_reference_metadata_from_batch,
)
from src.tasks.plcs.model_io.court_keypoint_checkpoint import (
    prepare_plcs_checkpoint_court_keypoint_config,
    validate_plcs_checkpoint_court_keypoints,
    write_plcs_checkpoint_court_keypoints,
)
from src.tasks.plcs.model_io.factory import (
    PLCSBoundModelIO,
    PLCSModelIOConfig,
    PLCSStandardBoundModelIO,
    bind_plcs_model_io,
    build_plcs_model_io,
)

__all__ = [
    "PLCSAdapter",
    "PLCSBoundModelIO",
    "PLCSDecodedPrediction",
    "PLCSInputProfile",
    "PLCSModelIOAdapter",
    "PLCSModelIOConfig",
    "PLCSPhysicalPrediction",
    "PLCSPreparedBatch",
    "PLCSReprojectionTarget",
    "PLCSReferenceMetadata",
    "PLCSStandardBoundModelIO",
    "bind_plcs_model_io",
    "build_plcs_model_io",
    "prepare_plcs_checkpoint_court_keypoint_config",
    "plcs_reference_metadata_from_batch",
    "validate_plcs_checkpoint_court_keypoints",
    "write_plcs_checkpoint_court_keypoints",
]
