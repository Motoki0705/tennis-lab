from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn.functional as F

from src.utils.models.components.ops.loader import require_time_local_cuda_extension
from src.utils.models.components.ops.time_local.layout import (
    build_sliding_window_layout,
    normalize_valid_mask,
)
from src.utils.models.components.ops.time_local.reference import (
    reference_time_local_attention,
)


class _CudaWindowGather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        extension = require_time_local_cuda_extension()
        windows = cast(
            torch.Tensor,
            extension.window_gather_forward(
                tensor.contiguous(),
                indices.contiguous(),
            ),
        )
        ctx.save_for_backward(indices.contiguous())
        ctx.input_shape = tuple(tensor.shape)
        return windows

    @staticmethod
    def backward(
        ctx: Any,
        grad_windows: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        (indices,) = cast(tuple[torch.Tensor], ctx.saved_tensors)
        batch_size, num_heads, seq_len, hidden_dim = ctx.input_shape
        _, _, _, window_size, _ = grad_windows.shape

        grad_flat = grad_windows.permute(0, 1, 4, 2, 3).reshape(
            batch_size * num_heads * hidden_dim,
            seq_len * window_size,
        )
        flat_indices = indices.reshape(1, seq_len * window_size).expand(
            batch_size * num_heads * hidden_dim,
            -1,
        )
        grad_input = grad_windows.new_zeros(batch_size * num_heads * hidden_dim, seq_len)
        grad_input.scatter_add_(1, flat_indices, grad_flat)
        grad_input = grad_input.reshape(batch_size, num_heads, hidden_dim, seq_len)
        grad_input = grad_input.permute(0, 1, 3, 2).contiguous()
        return grad_input, None


def cuda_time_local_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    window_radius: int,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, value must have shape [B, H, T, D]")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key, value must have the same shape")
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise ValueError("cuda_time_local_attention requires CUDA tensors")

    batch_size, num_heads, seq_len, head_dim = query.shape
    indices, index_valid = build_sliding_window_layout(
        seq_len=seq_len,
        window_radius=window_radius,
        device=query.device,
    )
    valid_mask = normalize_valid_mask(valid_mask)
    if valid_mask.device != query.device:
        valid_mask = valid_mask.to(query.device)

    local_keep = _build_local_keep_mask(valid_mask, indices, index_valid)
    can_use_cuda = local_keep.any(dim=-1).all(dim=-1)

    output = query.new_empty(query.shape)
    if can_use_cuda.any():
        keep_idx = can_use_cuda.nonzero(as_tuple=False).flatten()
        gathered_key = _cuda_window_gather(key.index_select(0, keep_idx), indices)
        gathered_value = _cuda_window_gather(value.index_select(0, keep_idx), indices)
        key_keep = _gather_valid_mask(valid_mask.index_select(0, keep_idx), indices) & index_valid.unsqueeze(0)

        local_batch = keep_idx.numel()
        window_size = gathered_key.shape[-2]

        query_local = query.index_select(0, keep_idx).permute(0, 2, 1, 3).reshape(
            local_batch * seq_len * num_heads,
            1,
            head_dim,
        )
        key_local = gathered_key.permute(0, 2, 1, 3, 4).reshape(
            local_batch * seq_len * num_heads,
            window_size,
            head_dim,
        )
        value_local = gathered_value.permute(0, 2, 1, 3, 4).reshape(
            local_batch * seq_len * num_heads,
            window_size,
            head_dim,
        )
        attn_mask = key_keep[:, :, None, :].expand(
            local_batch,
            seq_len,
            num_heads,
            window_size,
        ).reshape(local_batch * seq_len * num_heads, 1, window_size)

        local_output = F.scaled_dot_product_attention(
            query_local,
            key_local,
            value_local,
            attn_mask=attn_mask,
            dropout_p=dropout_p if training else 0.0,
            is_causal=False,
        )
        local_output = local_output.reshape(local_batch, seq_len, num_heads, head_dim)
        local_output = local_output.permute(0, 2, 1, 3).contiguous()
        output.index_copy_(0, keep_idx, local_output)

    fallback_mask = ~can_use_cuda
    if fallback_mask.any():
        fallback_idx = fallback_mask.nonzero(as_tuple=False).flatten()
        fallback_output = reference_time_local_attention(
            query.index_select(0, fallback_idx),
            key.index_select(0, fallback_idx),
            value.index_select(0, fallback_idx),
            valid_mask=valid_mask.index_select(0, fallback_idx),
            window_radius=window_radius,
            dropout_p=dropout_p,
            training=training,
        )
        output.index_copy_(0, fallback_idx, fallback_output)

    return output


def _cuda_window_gather(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return cast(torch.Tensor, _CudaWindowGather.apply(tensor, indices))


def _build_local_keep_mask(
    valid_mask: torch.Tensor,
    indices: torch.Tensor,
    index_valid: torch.Tensor,
) -> torch.Tensor:
    gathered_valid = _gather_valid_mask(valid_mask, indices)
    return gathered_valid & index_valid.unsqueeze(0)


def _gather_valid_mask(valid_mask: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = valid_mask.shape
    expanded_valid = valid_mask[:, None, :].expand(batch_size, seq_len, seq_len)
    expanded_indices = indices.unsqueeze(0).expand(batch_size, -1, -1)
    return torch.gather(expanded_valid, 2, expanded_indices)