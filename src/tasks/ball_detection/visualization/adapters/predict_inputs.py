"""Model-input adapters for ball-detection visualization.

Turns a loaded clip's model images into the overlapping sliding-window batches
the predictor consumes.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch

from src.utils.video.windows import (
    build_window_starts as build_window_starts,
)
from src.utils.video.windows import (
    chunked as chunked,
)


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
