"""Runtime dispatch helpers for MSDA forward execution."""

from __future__ import annotations

from torch import Tensor

from src.utils.models.components.ops.deformable.kernels.msda_fallback import ms_deform_attn_fallback


def run_msda_forward_or_fallback(
    *,
    ext: object | None,
    value: Tensor,
    spatial_shapes: Tensor,
    level_start_index: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    """Execute extension forward when available, otherwise run fallback."""
    if ext is not None and hasattr(ext, "ms_deform_attn_forward"):
        return ext.ms_deform_attn_forward(  # type: ignore[attr-defined]
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
        )

    return ms_deform_attn_fallback(
        value=value,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        sampling_locations=sampling_locations,
        attention_weights=attention_weights,
    )

