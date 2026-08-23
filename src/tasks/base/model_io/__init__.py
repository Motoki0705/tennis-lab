"""Canonical public API for shared task model I/O lifecycle contracts."""

from src.tasks.base.model_io.contracts import (
    BoundModelIO,
    ModelAdapterMismatchError,
    ModelArgument,
    ModelCall,
    ModelInputContractError,
    ModelIOAdapter,
    ModelIOContractError,
    ModelOutputContractError,
    bind_model_io,
)
from src.tasks.base.model_io.court_coordinate_contract import (
    CheckpointCourtCoordinateContract,
    extract_checkpoint_court_coordinate_normalization,
    resolve_checkpoint_court_coordinate_contract,
    validate_checkpoint_court_coordinate_contract,
    write_checkpoint_court_coordinate_contract,
)
from src.tasks.base.model_io.tensors import TensorSpec, require_tensor

__all__ = [
    "BoundModelIO",
    "CheckpointCourtCoordinateContract",
    "ModelAdapterMismatchError",
    "ModelArgument",
    "ModelCall",
    "ModelIOAdapter",
    "ModelIOContractError",
    "ModelInputContractError",
    "ModelOutputContractError",
    "TensorSpec",
    "bind_model_io",
    "extract_checkpoint_court_coordinate_normalization",
    "require_tensor",
    "resolve_checkpoint_court_coordinate_contract",
    "validate_checkpoint_court_coordinate_contract",
    "write_checkpoint_court_coordinate_contract",
]
