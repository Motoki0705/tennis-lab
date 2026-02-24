"""nn.Module wrapper for multi-scale deformable cross attention."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.utils.models.components.ops.deformable.api import multi_scale_deformable_attention
from src.utils.models.components.ops.deformable.config import MSDeformAttnConfig
from src.utils.models.components.ops.deformable.utils import build_level_start_index


class MultiScaleDeformableAttention(nn.Module):
    """Minimal module wrapper around deformable attention operator.

    This module is intentionally low-level and expects precomputed
    ``sampling_locations`` and ``attention_weights``.
    """

    def __init__(self, cfg: MSDeformAttnConfig) -> None:
        super().__init__()
        self.cfg = cfg

        if cfg.dim % cfg.num_heads != 0:
            raise ValueError("dim must be divisible by num_heads.")
        if cfg.num_levels <= 0 or cfg.num_points <= 0:
            raise ValueError("num_levels and num_points must be positive.")

        self.head_dim = cfg.dim // cfg.num_heads
        self.value_proj = nn.Linear(cfg.dim, cfg.dim)
        self.output_proj = nn.Linear(cfg.dim, cfg.dim)

    def forward(
        self,
        query: Tensor,
        value: Tensor,
        spatial_shapes: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
        *,
        level_start_index: Tensor | None = None,
    ) -> Tensor:
        """Run deformable attention.

        Args:
            query: (B, Q, D)
            value: (B, S, D)
            spatial_shapes: (L, 2)
            sampling_locations: (B, Q, H, L, P, 2)
            attention_weights: (B, Q, H, L, P)
            level_start_index: optional (L,)
        """
        bsz, _, dim = query.shape
        if dim != self.cfg.dim:
            raise ValueError(f"query dim mismatch: expected {self.cfg.dim}, got {dim}")

        value_proj = self.value_proj(value).view(bsz, value.shape[1], self.cfg.num_heads, self.head_dim)
        if level_start_index is None:
            level_start_index = build_level_start_index(spatial_shapes)

        out = multi_scale_deformable_attention(
            value=value_proj,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            sampling_locations=sampling_locations,
            attention_weights=attention_weights,
            prefer_cuda=self.cfg.use_cuda_kernel,
        )
        out = out.reshape(bsz, query.shape[1], self.cfg.dim)
        return self.output_proj(out)
