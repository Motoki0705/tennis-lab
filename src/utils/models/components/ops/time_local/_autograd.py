from __future__ import annotations

from types import ModuleType
from typing import Any, cast

import torch
import torch.nn.functional as F

from src.utils.models.components.ops.time_local.layout import (
    build_sliding_window_layout,
)


class _CudaWindowGather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
        indices: torch.Tensor,
        extension: ModuleType,
    ) -> torch.Tensor:
        windows = cast(
            torch.Tensor,
            extension.window_gather_forward(
                tensor.contiguous(),
                indices.contiguous(),
            ),
        )
        ctx.save_for_backward(indices.contiguous())
        ctx.input_shape = tensor.shape
        return windows

    @staticmethod
    def backward(
        ctx: Any,
        grad_windows: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
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
        grad_input = grad_windows.new_zeros(
            batch_size * num_heads * hidden_dim, seq_len
        )
        grad_input.scatter_add_(1, flat_indices, grad_flat)
        grad_input = grad_input.reshape(batch_size, num_heads, hidden_dim, seq_len)
        grad_input = grad_input.permute(0, 1, 3, 2).contiguous()
        return grad_input, None, None


def cuda_time_local_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: torch.Tensor,
    window_radius: int,
    dropout_p: float = 0.0,
    training: bool = False,
    extension: ModuleType,
) -> torch.Tensor:
    batch_size, num_heads, seq_len, head_dim = query.shape
    indices, index_valid = build_sliding_window_layout(
        seq_len=seq_len,
        window_radius=window_radius,
        device=query.device,
    )
    gathered_key = _cuda_window_gather(key, indices, extension)
    gathered_value = _cuda_window_gather(value, indices, extension)
    key_keep = _gather_attention_mask(attn_mask, indices) & index_valid.unsqueeze(0)
    window_size = gathered_key.shape[-2]

    query_local = query.permute(0, 2, 1, 3).reshape(
        batch_size * seq_len * num_heads,
        1,
        head_dim,
    )
    key_local = gathered_key.permute(0, 2, 1, 3, 4).reshape(
        batch_size * seq_len * num_heads,
        window_size,
        head_dim,
    )
    value_local = gathered_value.permute(0, 2, 1, 3, 4).reshape(
        batch_size * seq_len * num_heads,
        window_size,
        head_dim,
    )
    attn_mask = (
        key_keep[:, :, None, :]
        .expand(
            batch_size,
            seq_len,
            num_heads,
            window_size,
        )
        .reshape(batch_size * seq_len * num_heads, 1, window_size)
    )

    local_output = F.scaled_dot_product_attention(
        query_local,
        key_local,
        value_local,
        attn_mask=attn_mask,
        dropout_p=dropout_p if training else 0.0,
        is_causal=False,
    )
    local_output = local_output.reshape(batch_size, seq_len, num_heads, head_dim)
    return local_output.permute(0, 2, 1, 3).contiguous()


def _cuda_window_gather(
    tensor: torch.Tensor,
    indices: torch.Tensor,
    extension: ModuleType,
) -> torch.Tensor:
    return cast(torch.Tensor, _CudaWindowGather.apply(tensor, indices, extension))


def _gather_attention_mask(
    attn_mask: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    batch_size = attn_mask.shape[0]
    expanded_indices = indices.unsqueeze(0).expand(batch_size, -1, -1)
    return torch.gather(attn_mask, 2, expanded_indices)
