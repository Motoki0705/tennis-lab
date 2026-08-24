"""Lightning batch transfer that keeps frozen provenance metadata immutable."""

from __future__ import annotations

from typing import Protocol, TypeVar, cast, runtime_checkable

import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor

BatchT = TypeVar("BatchT")

_BLOCKING_DEVICE_TYPES = frozenset({"cpu", "mps"})


@runtime_checkable
class _TransferableDataType(Protocol):
    """Structural equivalent of Lightning Fabric's transferable batch leaf."""

    def to(self, device: torch.device, **kwargs: object) -> object: ...


def move_batch_to_device_preserving_frozen_metadata(
    batch: BatchT,
    device: torch.device | str,
) -> BatchT:
    """Move transferable leaves while treating frozen dataclasses as metadata.

    Lightning Fabric recursively reconstructs every dataclass during its
    default transfer.  That is invalid for frozen reference-selection and
    Court-frame provenance records.  ``allow_frozen=True`` retains those
    records as immutable CPU metadata while applying the normal ``.to(device)``
    protocol to the surrounding model/loss tensors.
    """
    resolved_device = torch.device(device)

    def move_leaf(value: _TransferableDataType) -> object:
        kwargs: dict[str, object] = {}
        if (
            isinstance(value, Tensor)
            and resolved_device.type not in _BLOCKING_DEVICE_TYPES
        ):
            kwargs["non_blocking"] = True
        moved = value.to(resolved_device, **kwargs)
        return value if moved is None else moved

    return cast(
        "BatchT",
        apply_to_collection(
            batch,
            dtype=_TransferableDataType,
            function=move_leaf,
            allow_frozen=True,
        ),
    )


__all__ = ["move_batch_to_device_preserving_frozen_metadata"]
