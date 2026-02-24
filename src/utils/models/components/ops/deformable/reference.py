"""Reference (readable) implementation helpers for deformable attention."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from src.utils.models.components.ops.deformable.utils import validate_msda_inputs


def ms_deform_attn_reference(
    value: Tensor,
    spatial_shapes: Tensor,
    level_start_index: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    """Compute multi-scale deformable attention using pure PyTorch ops.

    Shapes:
    - value: (B, S, H, D)
    - spatial_shapes: (L, 2)
    - level_start_index: (L,)
    - sampling_locations: (B, Q, H, L, P, 2) in [0,1]
    - attention_weights: (B, Q, H, L, P)

    Returns:
    - (B, Q, H, D)
    """
    validate_msda_inputs(value, spatial_shapes, level_start_index, sampling_locations, attention_weights)

    bsz, _, n_heads, head_dim = value.shape
    _, n_query, _, n_levels, n_points, _ = sampling_locations.shape

    out = value.new_zeros((bsz, n_query, n_heads, head_dim))
    for lvl in range(n_levels):
        h_l = int(spatial_shapes[lvl, 0].item())
        w_l = int(spatial_shapes[lvl, 1].item())
        start = int(level_start_index[lvl].item())
        end = start + h_l * w_l

        value_l = value[:, start:end]  # (B, Hl*Wl, H, D)
        value_l = value_l.permute(0, 2, 3, 1).contiguous().reshape(bsz * n_heads, head_dim, h_l, w_l)

        grid = sampling_locations[:, :, :, lvl]  # (B,Q,H,P,2)
        grid = grid.permute(0, 2, 1, 3, 4).contiguous().reshape(bsz * n_heads, n_query * n_points, 1, 2)
        grid = grid * 2.0 - 1.0

        sampled = F.grid_sample(
            input=value_l,
            grid=grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.view(bsz, n_heads, head_dim, n_query, n_points)
        sampled = sampled.permute(0, 3, 1, 4, 2).contiguous()  # (B,Q,H,P,D)

        w = attention_weights[:, :, :, lvl, :].unsqueeze(-1)
        out = out + (sampled * w).sum(dim=3)

    return out
