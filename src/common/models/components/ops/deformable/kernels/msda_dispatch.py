"""Dispatch deformable attention to CUDA extension or PyTorch fallback."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.cpp_extension import load

from src.common.models.components.ops.deformable.kernels.msda_fallback import ms_deform_attn_fallback

_MSDA_EXT = None
_MSDA_EXT_LOAD_ATTEMPTED = False


def get_msda_extension():
    """Load and cache CUDA extension module for MSDA kernels if available."""
    global _MSDA_EXT, _MSDA_EXT_LOAD_ATTEMPTED
    if _MSDA_EXT_LOAD_ATTEMPTED:
        return _MSDA_EXT
    _MSDA_EXT_LOAD_ATTEMPTED = True

    if os.environ.get("MSDA_FORCE_FALLBACK", "0") == "1":
        return None
    if not torch.cuda.is_available():
        return None

    csrc_dir = Path(__file__).resolve().parent.parent / "csrc"
    sources = [
        str(csrc_dir / "binding.cpp"),
        str(csrc_dir / "deformable_cuda.cpp"),
        str(csrc_dir / "deformable_cuda_kernel.cu"),
    ]

    try:
        _MSDA_EXT = load(
            name="msda_ops",
            sources=sources,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception:
        _MSDA_EXT = None
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
    if ext is None:
        return ms_deform_attn_fallback(
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            sampling_locations=sampling_locations,
            attention_weights=attention_weights,
        )

    if hasattr(ext, "ms_deform_attn_forward"):
        return ext.ms_deform_attn_forward(
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
