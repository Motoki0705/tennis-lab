"""Sequence helpers for cropping and masking."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor


def build_valid_mask(T: int, seq_len: Tensor) -> Tensor:
    """Build a valid-length mask for a sequence or batch.

    Args:
        T: Total sequence length.
        seq_len: Scalar tensor or (B,) tensor of lengths.

    Returns:
        Boolean mask of shape (T,) or (B, T).
    """
    if seq_len.dim() == 0:
        t = torch.arange(T, device=seq_len.device)
        return t < seq_len.to(torch.long)
    t = torch.arange(T, device=seq_len.device)[None, :]
    return t < seq_len.to(torch.long).view(-1, 1)


def crop_to_max_len(
    tensors: dict[str, Tensor],
    *,
    seq_len: int,
    max_seq_len: int,
    mode: Literal["random", "center"] = "random",
    rng: torch.Generator | None = None,
) -> tuple[dict[str, Tensor], int]:
    """Crop temporal tensors to max_seq_len with consistent offsets.

    Args:
        tensors: Mapping of name to (T, ...) tensors to crop.
        seq_len: Valid sequence length.
        max_seq_len: Maximum allowed sequence length.
        mode: "random" or "center" crop selection.
        rng: Optional torch RNG for reproducibility.

    Returns:
        (cropped tensors, new_seq_len)
    """
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}.")
    first = next(iter(tensors.values()))
    T = int(first.shape[0])
    if T <= max_seq_len:
        return tensors, min(seq_len, T)

    crop_len = max_seq_len
    max_start = max(0, seq_len - crop_len)
    if mode == "random" and max_start > 0:
        start = int(torch.randint(0, max_start + 1, (1,), generator=rng).item())
    else:
        start = max_start // 2
    end = start + crop_len

    cropped = {k: v[start:end] for k, v in tensors.items()}
    new_seq_len = max(0, min(seq_len - start, crop_len))
    return cropped, new_seq_len
