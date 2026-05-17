"""Batching helpers for temporal video inference."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch

from src.utils.video.types import TemporalBatch, TemporalWindow


def iter_temporal_batches(
    windows: Iterable[TemporalWindow[torch.Tensor]],
    *,
    batch_size: int,
    pin_memory: bool = False,
) -> Iterator[TemporalBatch]:
    """Stack temporal windows into ``(B, T, C, H, W)`` tensors."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    pending: list[TemporalWindow[torch.Tensor]] = []
    for window in windows:
        pending.append(window)
        if len(pending) == batch_size:
            yield _stack_batch(pending, pin_memory=pin_memory)
            pending = []

    if pending:
        yield _stack_batch(pending, pin_memory=pin_memory)


def _stack_batch(
    windows: list[TemporalWindow[torch.Tensor]],
    *,
    pin_memory: bool,
) -> TemporalBatch:
    sequences = [torch.stack(window.frames, dim=0) for window in windows]
    tensor = torch.stack(sequences, dim=0)
    if pin_memory and torch.cuda.is_available():
        tensor = tensor.pin_memory()
    return TemporalBatch(windows=tuple(windows), tensor=tensor)
