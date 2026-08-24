"""Canonical public PLCS model I/O composition API."""

from src.tasks.plcs.model_io.adapters import (
    PLCSAdapter,
    PLCSModelIOAdapter,
    PLCSTrackQueryIOAdapter,
    PLCSTrackQueryReferenceIOAdapter,
)
from src.tasks.plcs.model_io.contracts import (
    PLCSDecodedPrediction,
    PLCSInputProfile,
    PLCSPhysicalPrediction,
    PLCSPreparedBatch,
    PLCSReferenceMetadata,
    PLCSReprojectionTarget,
    PLCSTrackingDecodedPrediction,
    plcs_reference_metadata_from_batch,
)
from src.tasks.plcs.model_io.court_keypoint_checkpoint import (
    prepare_plcs_checkpoint_court_keypoint_config,
    validate_plcs_checkpoint_court_keypoints,
    write_plcs_checkpoint_court_keypoints,
)
from src.tasks.plcs.model_io.factory import (
    PLCSBoundModelIO,
    PLCSStandardBoundModelIO,
    PLCSTrackingBoundModelIO,
    bind_plcs_model_io,
    build_plcs_model_io,
)
from src.tasks.plcs.model_io.track_query_reference_checkpoint import (
    resolve_plcs_track_query_reference_contract,
    validate_plcs_checkpoint_track_query_reference,
    write_plcs_checkpoint_track_query_reference,
)

__all__ = [
    "PLCSAdapter",
    "PLCSBoundModelIO",
    "PLCSDecodedPrediction",
    "PLCSInputProfile",
    "PLCSModelIOAdapter",
    "PLCSPhysicalPrediction",
    "PLCSPreparedBatch",
    "PLCSReprojectionTarget",
    "PLCSReferenceMetadata",
    "PLCSStandardBoundModelIO",
    "PLCSTrackQueryIOAdapter",
    "PLCSTrackQueryReferenceIOAdapter",
    "PLCSTrackingBoundModelIO",
    "PLCSTrackingDecodedPrediction",
    "bind_plcs_model_io",
    "build_plcs_model_io",
    "prepare_plcs_checkpoint_court_keypoint_config",
    "plcs_reference_metadata_from_batch",
    "resolve_plcs_track_query_reference_contract",
    "validate_plcs_checkpoint_court_keypoints",
    "validate_plcs_checkpoint_track_query_reference",
    "write_plcs_checkpoint_court_keypoints",
    "write_plcs_checkpoint_track_query_reference",
]
