"""Memory controls with no parameters or changes to the temporal contract."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

Result = TypeVar("Result")


def checkpoint_call(function: Callable[..., Result], *args: Tensor, enabled: bool) -> Result:
    if enabled and torch.is_grad_enabled():
        return cast(Result, checkpoint(function, *args, use_reentrant=False))
    return function(*args)


def decode_frame_heatmaps(
    decoder: nn.Module, head: nn.Module, features: tuple[Tensor, ...], *,
    output_size: tuple[int, int], frame_chunk_size: int | None, activation_checkpointing: bool,
) -> Tensor:
    """Decode frame-local DPT in bounded groups, checkpointing through the head.

    Checkpointing the complete decode avoids retaining the wide /4 feature map.
    Frame-local chunks bound recomputation while retaining only their input levels.
    CNN halo frames have already been discarded and are never decoded twice.
    """
    batch, frames = features[0].shape[:2]

    def decode(*levels: Tensor) -> Tensor:
        fused = decoder([level.flatten(0, 1) for level in levels])
        return cast(Tensor, head(F.interpolate(fused, size=output_size, mode="bilinear", align_corners=False)))

    chunk_size = frames if frame_chunk_size is None else frame_chunk_size
    chunks: list[Tensor] = []
    for start in range(0, frames, chunk_size):
        stop = min(start + chunk_size, frames)
        logits = checkpoint_call(decode, *(level[:, start:stop] for level in features),
                                 enabled=activation_checkpointing and decoder.training)
        chunks.append(logits.reshape(batch, stop - start, *output_size))
    return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=1)
