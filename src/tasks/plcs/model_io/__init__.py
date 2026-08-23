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
    PLCSTrackingDecodedPrediction,
)
from src.tasks.plcs.model_io.court_coordinate_checkpoint import (
    load_plcs_checkpoint_mapping,
    prepare_plcs_checkpoint_config,
    validate_plcs_checkpoint_normalization,
    write_plcs_checkpoint_normalization,
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
    "PLCSStandardBoundModelIO",
    "PLCSTrackQueryIOAdapter",
    "PLCSTrackingBoundModelIO",
    "PLCSTrackingDecodedPrediction",
    "bind_plcs_model_io",
    "build_plcs_model_io",
    "load_plcs_checkpoint_mapping",
    "prepare_plcs_checkpoint_config",
    "validate_plcs_checkpoint_normalization",
    "write_plcs_checkpoint_normalization",
]
