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
    "TensorSpec",
    "bind_model_io",
    "require_tensor",
]
