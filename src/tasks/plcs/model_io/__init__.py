"""Canonical public PLCS model I/O composition API."""

from src.tasks.plcs.model_io.adapters import (
    PLCSAdapter,
    PLCSModelIOAdapter,
    PLCSTrackQueryIOAdapter,
)
from src.tasks.plcs.model_io.contracts import (
    PLCSDecodedPrediction,
    PLCSInputProfile,
    PLCSPhysicalPrediction,
    PLCSPreparedBatch,
    PLCSReprojectionTarget,
    PLCSTrackingDecodedPrediction,
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

__all__ = [
    "PLCSAdapter",
    "PLCSBoundModelIO",
    "PLCSDecodedPrediction",
    "PLCSInputProfile",
    "PLCSModelIOAdapter",
    "PLCSPhysicalPrediction",
    "PLCSPreparedBatch",
    "PLCSReprojectionTarget",
    "PLCSStandardBoundModelIO",
    "PLCSTrackQueryIOAdapter",
    "PLCSTrackingBoundModelIO",
    "PLCSTrackingDecodedPrediction",
    "bind_plcs_model_io",
    "build_plcs_model_io",
    "prepare_plcs_checkpoint_court_keypoint_config",
    "validate_plcs_checkpoint_court_keypoints",
    "write_plcs_checkpoint_court_keypoints",
]
