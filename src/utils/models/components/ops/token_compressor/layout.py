"""Deterministic previous/current source layout for token compression."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class TokenCompressorLayout:
    """Data-independent ``[Tc, 2 * ratio]`` source layout."""

    source_indices: Tensor
    source_branches: Tensor
    boundary_valid: Tensor


def build_token_compressor_layout(
    sequence_length: int,
    compression_ratio: int,
    device: torch.device,
) -> TokenCompressorLayout:
    """Build previous/current source indices without a dense time mask."""
    if type(sequence_length) is not int or sequence_length <= 0:
        raise ValueError(
            f"sequence_length must be a positive int, got {sequence_length!r}"
        )
    if type(compression_ratio) is not int or compression_ratio < 2:
        raise ValueError(
            f"compression_ratio must be an int of at least 2, got {compression_ratio!r}"
        )
    compressed_length = (sequence_length + compression_ratio - 1) // compression_ratio
    offsets = torch.arange(compression_ratio, device=device)
    current_starts = torch.arange(compressed_length, device=device)
    current_starts = current_starts * compression_ratio
    current = current_starts[:, None] + offsets[None, :]
    previous = current - compression_ratio
    source_indices = torch.cat((previous, current), dim=1)
    boundary_valid = (source_indices >= 0) & (source_indices < sequence_length)
    safe_indices = source_indices.clamp(min=0, max=sequence_length - 1)
    source_branches = torch.cat(
        (torch.zeros_like(previous), torch.ones_like(current)), dim=1
    )
    return TokenCompressorLayout(
        source_indices=safe_indices,
        source_branches=source_branches,
        boundary_valid=boundary_valid,
    )


__all__ = ["TokenCompressorLayout", "build_token_compressor_layout"]
