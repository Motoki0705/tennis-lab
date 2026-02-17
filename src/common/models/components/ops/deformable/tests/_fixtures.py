"""Shared fixtures/helpers for deformable op tests."""

from __future__ import annotations

import os
import time
from collections.abc import Callable

import torch


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def make_msda_case(
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    n_query: int,
    n_heads: int,
    head_dim: int,
    spatial_shapes_hw: list[tuple[int, int]],
    n_points: int,
    requires_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build deterministic random tensors for MSDA tests."""
    torch.manual_seed(1234)

    spatial_shapes = torch.tensor(spatial_shapes_hw, device=device, dtype=torch.long)
    n_levels = int(spatial_shapes.shape[0])
    total_tokens = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item())

    level_start_index = torch.zeros(n_levels, device=device, dtype=torch.long)
    if n_levels > 1:
        level_start_index[1:] = torch.cumsum(spatial_shapes[:-1, 0] * spatial_shapes[:-1, 1], dim=0)

    value = torch.randn(
        batch_size,
        total_tokens,
        n_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    sampling_locations = torch.rand(
        batch_size,
        n_query,
        n_heads,
        n_levels,
        n_points,
        2,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    attention_weights = torch.rand(
        batch_size,
        n_query,
        n_heads,
        n_levels,
        n_points,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    attention_weights = attention_weights / (attention_weights.sum(dim=(-1, -2), keepdim=True) + 1e-6)
    if requires_grad:
        attention_weights.retain_grad()

    return value, spatial_shapes, level_start_index, sampling_locations, attention_weights


def cuda_step_time_ms(fn: Callable[[], torch.Tensor], warmup: int = 2, iters: int = 5) -> float:
    """Measure average CUDA elapsed time in milliseconds."""
    for _ in range(max(0, warmup)):
        _ = fn()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(max(1, iters)):
        _ = fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / float(max(1, iters))


def cuda_peak_mem_mb(fn: Callable[[], torch.Tensor]) -> float:
    """Measure CUDA peak memory for a callable."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _ = fn()
    torch.cuda.synchronize()
    return float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
