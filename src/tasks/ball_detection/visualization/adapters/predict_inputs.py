"""Model-input adapters for ball-detection visualization.

Turns a loaded clip's model images into the overlapping sliding-window batches
the predictor consumes.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch


def build_window_starts(
    *,
    frame_count: int,
    sequence_length: int,
    stride: int,
) -> list[int]:
    """Compute sliding-window start indices covering every frame.

    The final window is always anchored at ``frame_count - sequence_length`` so
    the tail of the clip is covered even when ``stride`` does not divide evenly.
    """
    if frame_count < sequence_length:
        raise ValueError(
            f"frame_count must be >= sequence_length, got {frame_count} < {sequence_length}."
        )

    starts = list(range(0, frame_count - sequence_length + 1, stride))
    last_start = frame_count - sequence_length
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def chunked(values: Sequence[int], chunk_size: int) -> Iterator[list[int]]:
    """Yield ``values`` in lists of at most ``chunk_size`` items."""
    for start_index in range(0, len(values), chunk_size):
        yield list(values[start_index : start_index + chunk_size])


def iter_window_batches(
    *,
    model_images: torch.Tensor,
    window_starts: Sequence[int],
    sequence_length: int,
    batch_size: int,
) -> Iterator[tuple[list[int], torch.Tensor]]:
    """Yield ``(start_indices, batch)`` for each chunk of sliding windows.

    Each ``batch`` has shape ``(B, sequence_length, C, H, W)`` built by stacking
    the windows that start at the indices in ``start_indices``.
    """
    for start_chunk in chunked(window_starts, batch_size):
        batch = torch.stack(
            [model_images[start : start + sequence_length] for start in start_chunk],
            dim=0,
        )
        yield start_chunk, batch
