from __future__ import annotations

import os

import torch

from src.utils.models.components.ops.loader import get_time_local_cuda_extension
from src.utils.models.components.ops.time_local.reference import (
    reference_time_local_attention,
)


def time_local_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    window_radius: int,
    dropout_p: float = 0.0,
    training: bool = False,
    use_cuda: bool | None = None,
) -> torch.Tensor:
    if _should_use_cuda(query, use_cuda):
        from src.utils.models.components.ops.time_local._autograd import (  # pragma: no cover - optional CUDA path
            cuda_time_local_attention,
        )

        return cuda_time_local_attention(
            query,
            key,
            value,
            valid_mask=valid_mask,
            window_radius=window_radius,
            dropout_p=dropout_p,
            training=training,
        )
    return reference_time_local_attention(
        query,
        key,
        value,
        valid_mask=valid_mask,
        window_radius=window_radius,
        dropout_p=dropout_p,
        training=training,
    )


def _should_use_cuda(tensor: torch.Tensor, use_cuda: bool | None) -> bool:
    force_reference = os.environ.get("TENNIS_LAB_FORCE_TIME_LOCAL_REFERENCE", "") in {
        "1",
        "true",
        "yes",
        "on",
    }
    prefer_cuda = os.environ.get("TENNIS_LAB_USE_TIME_LOCAL_CUDA", "") in {
        "1",
        "true",
        "yes",
        "on",
    }
    if use_cuda is False or force_reference:
        if use_cuda is True and force_reference:
            raise RuntimeError("TENNIS_LAB_FORCE_TIME_LOCAL_REFERENCE is set")
        return False
    if use_cuda is None and not prefer_cuda:
        return False
    if not tensor.is_cuda:
        if use_cuda is True:
            raise RuntimeError("use_cuda=True requires CUDA tensors")
        return False
    if get_time_local_cuda_extension() is None:
        if use_cuda is True:
            raise RuntimeError("Time-local CUDA extension is not available")
        return False
    return True