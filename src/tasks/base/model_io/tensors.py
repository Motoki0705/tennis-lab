"""Generic tensor validation for task model I/O boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from src.tasks.base.model_io.contracts import (
    ModelInputContractError,
    ModelIOContractError,
)


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """Task-independent dtype, rank, and fixed-dimension tensor constraints.

    ``None`` dimensions are unconstrained. Task-specific axis meanings and
    cross-tensor invariants belong in the task adapter that uses this spec.
    """

    shape: tuple[int | None, ...] | None = None
    dtypes: frozenset[torch.dtype] | None = None

    def __post_init__(self) -> None:
        if self.shape is not None and any(
            dimension is not None and dimension < 0 for dimension in self.shape
        ):
            raise ModelIOContractError(
                "TensorSpec fixed dimensions must be non-negative or None."
            )
        if self.dtypes is not None and not self.dtypes:
            raise ModelIOContractError("TensorSpec dtypes must be non-empty or None.")

    @property
    def rank(self) -> int | None:
        """Return the required rank when a shape contract is present."""
        return None if self.shape is None else len(self.shape)

    def validate(self, name: str, value: object) -> Tensor:
        """Return ``value`` as a tensor or raise an explicit boundary error."""
        if not isinstance(value, Tensor):
            raise ModelInputContractError(
                f"{name} must be a torch.Tensor, got {type(value).__name__}."
            )
        if self.dtypes is not None and value.dtype not in self.dtypes:
            expected = ", ".join(sorted(str(dtype) for dtype in self.dtypes))
            raise ModelInputContractError(
                f"{name} must use one of ({expected}), got {value.dtype}."
            )
        if self.shape is None:
            return value
        if value.ndim != len(self.shape):
            raise ModelInputContractError(
                f"{name} must have rank {len(self.shape)}, got shape "
                f"{tuple(value.shape)}."
            )
        mismatches = [
            (axis, expected, actual)
            for axis, (expected, actual) in enumerate(
                zip(self.shape, value.shape, strict=True)
            )
            if expected is not None and expected != actual
        ]
        if mismatches:
            details = ", ".join(
                f"axis {axis}: expected {expected}, got {actual}"
                for axis, expected, actual in mismatches
            )
            raise ModelInputContractError(f"{name} shape mismatch ({details}).")
        return value


def require_tensor(
    batch: Mapping[str, object],
    name: str,
    *,
    spec: TensorSpec | None = None,
) -> Tensor:
    """Read and validate a required tensor field from a batch mapping."""
    if name not in batch:
        raise ModelInputContractError(f"Required model input {name!r} is missing.")
    return (spec or TensorSpec()).validate(name, batch[name])


__all__ = ["TensorSpec", "require_tensor"]
