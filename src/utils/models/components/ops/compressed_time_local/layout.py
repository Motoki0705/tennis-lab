"""Index layouts for attention over fixed windows of compressed time states."""

from __future__ import annotations

import torch
from torch import Tensor


def _require_positive_int(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")


def build_compressed_sliding_window_layout(
    *,
    query_len: int,
    key_len: int,
    compression_ratio: int,
    window_radius: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Return clamped compressed-key indices and their boundary-valid mask.

    The returned tensors both have shape ``[query_len, 2 * window_radius + 1]``.
    ``indices`` has dtype ``long`` and is safe to gather with; entries that were
    outside ``[0, key_len)`` are clamped but are marked false in ``index_valid``.
    """
    _require_positive_int("query_len", query_len)
    _require_positive_int("key_len", key_len)
    if type(compression_ratio) is not int or compression_ratio < 2:
        raise ValueError(
            f"compression_ratio must be an int of at least 2, got {compression_ratio!r}"
        )
    if type(window_radius) is not int or window_radius < 0:
        raise ValueError(
            f"window_radius must be a non-negative int, got {window_radius!r}"
        )
    if not isinstance(device, torch.device):
        raise TypeError(f"device must be torch.device, got {type(device).__name__}")

    expected_key_len = (query_len + compression_ratio - 1) // compression_ratio
    if key_len != expected_key_len:
        raise ValueError(
            "key_len must equal ceil(query_len / compression_ratio): "
            f"expected {expected_key_len}, got {key_len}"
        )

    query_positions = torch.arange(query_len, device=device, dtype=torch.long)
    centers = torch.div(query_positions, compression_ratio, rounding_mode="floor")
    offsets = torch.arange(
        -window_radius,
        window_radius + 1,
        device=device,
        dtype=torch.long,
    )
    raw_indices = centers[:, None] + offsets[None, :]
    index_valid = (raw_indices >= 0) & (raw_indices < key_len)
    indices = raw_indices.clamp(min=0, max=key_len - 1)
    return indices, index_valid
