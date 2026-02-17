"""Dispatch deformable attention to CUDA extension or PyTorch fallback."""

from __future__ import annotations

from torch import Tensor

from src.common.models.components.ops.deformable.kernels.msda_ext_loader import try_load_msda_extension
from src.common.models.components.ops.deformable.kernels.msda_runtime import run_msda_forward_or_fallback

_MSDA_EXT = None
_MSDA_EXT_LOAD_ATTEMPTED = False


def get_msda_extension():
    """Load and cache CUDA extension module for MSDA kernels if available."""
    global _MSDA_EXT, _MSDA_EXT_LOAD_ATTEMPTED
    _MSDA_EXT, _MSDA_EXT_LOAD_ATTEMPTED = try_load_msda_extension(
        already_attempted=_MSDA_EXT_LOAD_ATTEMPTED,
        cached_ext=_MSDA_EXT,
    )
    return _MSDA_EXT


def ms_deform_attn_dispatch(
    value: Tensor,
    spatial_shapes: Tensor,
    level_start_index: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
    *,
    prefer_cuda: bool = True,
) -> Tensor:
    """Dispatch forward path to CUDA extension when available, otherwise fallback."""
    ext = get_msda_extension() if prefer_cuda else None
    return run_msda_forward_or_fallback(
        ext=ext,
        value=value,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        sampling_locations=sampling_locations,
        attention_weights=attention_weights,
    )
