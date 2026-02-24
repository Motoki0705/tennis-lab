"""Public API functions for deformable attention ops."""

from __future__ import annotations

from torch import Tensor

from src.utils.models.components.ops.deformable.kernels import ms_deform_attn
from src.utils.models.components.ops.deformable.utils import build_level_start_index


def multi_scale_deformable_attention(
    value: Tensor,
    spatial_shapes: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
    *,
    level_start_index: Tensor | None = None,
    prefer_cuda: bool = True,
) -> Tensor:
    """Compute multi-scale deformable attention output.

    If ``level_start_index`` is omitted, it is derived from ``spatial_shapes``.
    """
    if level_start_index is None:
        level_start_index = build_level_start_index(spatial_shapes)
    return ms_deform_attn(
        value=value,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        sampling_locations=sampling_locations,
        attention_weights=attention_weights,
        prefer_cuda=prefer_cuda,
    )
