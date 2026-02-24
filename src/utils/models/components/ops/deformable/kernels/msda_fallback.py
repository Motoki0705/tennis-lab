"""PyTorch fallback kernel entrypoint for deformable attention."""

from __future__ import annotations

from torch import Tensor

from src.utils.models.components.ops.deformable.reference import ms_deform_attn_reference


def ms_deform_attn_fallback(
    value: Tensor,
    spatial_shapes: Tensor,
    level_start_index: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    """Fallback implementation used when compiled kernels are unavailable."""
    return ms_deform_attn_reference(
        value=value,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        sampling_locations=sampling_locations,
        attention_weights=attention_weights,
    )
