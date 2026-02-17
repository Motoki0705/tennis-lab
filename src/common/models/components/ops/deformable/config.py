"""Configuration objects for deformable attention ops/modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MSDeformAttnConfig:
    """Runtime configuration for multi-scale deformable attention."""

    dim: int
    num_heads: int
    num_levels: int
    num_points: int
    im2col_step: int = 64
    use_cuda_kernel: bool = True
    allow_fallback: bool = True
