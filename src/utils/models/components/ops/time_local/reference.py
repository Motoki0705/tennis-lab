from __future__ import annotations

import torch
import torch.nn.functional as F


def reference_time_local_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: torch.Tensor,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask[:, None, :, :],
        dropout_p=dropout_p if training else 0.0,
        is_causal=False,
    )
