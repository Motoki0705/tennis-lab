from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Literal

import torch

from src.utils.models.components.ops.loader import require_time_local_cuda_extension
from src.utils.models.components.ops.time_local._autograd import (
    cuda_time_local_attention,
)
from src.utils.models.components.ops.time_local.reference import (
    reference_time_local_attention,
)

TimeLocalAttentionExecutor = Callable[..., torch.Tensor]


def resolve_time_local_attention(
    backend: Literal["reference", "cuda"],
    *,
    window_radius: int,
) -> TimeLocalAttentionExecutor:
    """Resolve one time-local implementation at construction/composition time."""
    if type(window_radius) is not int or window_radius < 0:
        raise ValueError(
            f"window_radius must be a non-negative int, got {window_radius!r}."
        )
    if backend == "reference":
        return reference_time_local_attention
    if backend == "cuda":
        extension = require_time_local_cuda_extension()
        return partial(
            cuda_time_local_attention,
            window_radius=window_radius,
            extension=extension,
        )
    raise ValueError(f"Unsupported time-local backend: {backend!r}.")
