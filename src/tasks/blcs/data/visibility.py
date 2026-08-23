"""Visibility-aware normalization for persisted BLCS image observations."""

from __future__ import annotations

from torch import Tensor


def zero_invisible_uv(uv: Tensor, visibility: Tensor) -> Tensor:
    """Return UV coordinates with every invisible observation replaced by zero.

    Persisted projections may retain finite or non-finite off-screen coordinates
    for invisible points.  Those values are not model observations and must not
    cross the dataset boundary.  Visible values are deliberately left unchanged
    so the strict model-input adapter can still reject malformed coordinates.
    """
    if uv.ndim == 0 or uv.shape[-1] != 2:
        raise ValueError(f"uv must end in a coordinate axis of size 2; got {uv.shape}.")
    if visibility.shape != uv.shape[:-1]:
        raise ValueError(
            "visibility shape must match uv without the coordinate axis; "
            f"got visibility={visibility.shape}, uv={uv.shape}."
        )
    invisible = ~visibility.bool()
    return uv.masked_fill(invisible.unsqueeze(-1), 0.0)


__all__ = ["zero_invisible_uv"]
