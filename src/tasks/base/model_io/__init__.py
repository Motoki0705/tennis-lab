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
from src.tasks.base.model_io.court_keypoint_contract import (
    ModelArtifactCourtKeypointContract,
    extract_model_artifact_court_keypoint_contract,
    resolve_model_artifact_court_keypoint_contract,
    validate_model_artifact_court_keypoint_contract,
    write_model_artifact_court_keypoint_contract,
)
from src.tasks.base.model_io.tensors import TensorSpec, require_tensor

__all__ = [
    "BoundModelIO",
    "ModelAdapterMismatchError",
    "ModelArgument",
    "ModelCall",
    "ModelIOAdapter",
    "ModelIOContractError",
    "ModelInputContractError",
    "ModelOutputContractError",
    "ModelArtifactCourtKeypointContract",
    "TensorSpec",
    "bind_model_io",
    "extract_model_artifact_court_keypoint_contract",
    "require_tensor",
    "resolve_model_artifact_court_keypoint_contract",
    "validate_model_artifact_court_keypoint_contract",
    "write_model_artifact_court_keypoint_contract",
]
