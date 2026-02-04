"""Shared collate utilities for padded sequence batches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor


def collate_padded_batch(
    batch: Sequence[Mapping[str, Tensor]],
    *,
    sequence_keys: Sequence[str],
    static_keys: Sequence[str] | None = None,
    seq_len_key: str = "seq_len",
    mask_key: str | None = None,
) -> dict[str, Tensor]:
    """Collate a batch of samples with padded temporal sequences.

    Args:
        batch: List of sample dictionaries.
        sequence_keys: Keys for temporal tensors (shape (T, ...)).
        static_keys: Keys for per-sample tensors (shape (...)).
        seq_len_key: Key holding sequence length (scalar tensor or int).
        mask_key: Optional key name for the output padding mask (1=valid).

    Returns:
        Dictionary containing padded tensors and sequence lengths.
    """
    if not batch:
        raise ValueError("Batch must contain at least one sample.")
    static_keys = list(static_keys or [])

    seq_lens = torch.tensor(
        [
            int(s[seq_len_key].item())
            if isinstance(s[seq_len_key], Tensor)
            else int(s[seq_len_key])
            for s in batch
        ],
        dtype=torch.long,
    )
    max_len = int(seq_lens.max().item()) if seq_lens.numel() else 0

    out: dict[str, Tensor] = {seq_len_key: seq_lens}
    for key in sequence_keys:
        sample = batch[0][key]
        shape = sample.shape[1:]
        out[key] = torch.zeros(len(batch), max_len, *shape, dtype=sample.dtype)

    for key in static_keys:
        sample = batch[0][key]
        shape = sample.shape
        out[key] = torch.zeros(len(batch), *shape, dtype=sample.dtype)

    if mask_key is not None:
        out[mask_key] = torch.zeros(len(batch), max_len, dtype=torch.float32)

    for i, sample in enumerate(batch):
        seq_len = int(seq_lens[i].item())
        for key in sequence_keys:
            out[key][i, :seq_len] = sample[key][:seq_len]
        for key in static_keys:
            out[key][i] = sample[key]
        if mask_key is not None:
            out[mask_key][i, :seq_len] = 1.0

    return out
