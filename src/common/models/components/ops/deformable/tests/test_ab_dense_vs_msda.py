"""A/B tests: dense attention baseline vs MSDA on speed/memory/trainability."""

from __future__ import annotations

import time

import pytest
import torch
from torch import nn

from src.common.models.components.ops.deformable.api import multi_scale_deformable_attention
from src.common.models.components.ops.deformable.kernels.msda_dispatch import get_msda_extension
from src.common.models.components.ops.deformable.tests._fixtures import env_float, make_msda_case


class _DenseCrossAttnBaseline(nn.Module):
    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.dim = dim
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)

    def forward(self, query: torch.Tensor, value_flat: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(query, value_flat, value_flat, need_weights=False)
        return out


@pytest.mark.cuda
@pytest.mark.slow
def test_ab_dense_vs_msda_cuda_speed_memory() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    ext = get_msda_extension()
    if ext is None:
        pytest.skip("MSDA CUDA extension is not available.")

    speed_ratio_max = env_float("MSDA_AB_MAX_SPEED_RATIO", 1.5)
    mem_ratio_max = env_float("MSDA_AB_MAX_MEM_RATIO", 1.2)

    device = torch.device("cuda")
    bsz, n_query, n_heads, head_dim = 2, 128, 8, 16
    dim = n_heads * head_dim

    value, spatial_shapes, level_start_index, sampling_locations, attention_weights = make_msda_case(
        device=device,
        dtype=torch.float32,
        batch_size=bsz,
        n_query=n_query,
        n_heads=n_heads,
        head_dim=head_dim,
        spatial_shapes_hw=[(24, 32), (12, 16)],
        n_points=4,
        requires_grad=False,
    )
    value_flat = value.reshape(bsz, value.shape[1], dim)
    query = torch.randn(bsz, n_query, dim, device=device)

    dense = _DenseCrossAttnBaseline(dim=dim, n_heads=n_heads).to(device)

    def run_dense() -> torch.Tensor:
        return dense(query, value_flat)

    def run_msda() -> torch.Tensor:
        out = multi_scale_deformable_attention(
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            sampling_locations=sampling_locations,
            attention_weights=attention_weights,
            prefer_cuda=True,
        )
        return out.reshape(bsz, n_query, dim)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _ = run_dense()
    torch.cuda.synchronize()
    dense_mem = float(torch.cuda.max_memory_allocated())

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _ = run_msda()
    torch.cuda.synchronize()
    msda_mem = float(torch.cuda.max_memory_allocated())

    for _ in range(2):
        _ = run_dense()
        _ = run_msda()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(5):
        _ = run_dense()
    torch.cuda.synchronize()
    dense_ms = (time.perf_counter() - t0) * 1000.0 / 5.0

    t0 = time.perf_counter()
    for _ in range(5):
        _ = run_msda()
    torch.cuda.synchronize()
    msda_ms = (time.perf_counter() - t0) * 1000.0 / 5.0

    assert msda_mem <= dense_mem * mem_ratio_max
    assert msda_ms <= dense_ms * speed_ratio_max


@pytest.mark.cuda
def test_ab_dense_vs_msda_toy_trainability() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    ext = get_msda_extension()
    if ext is None:
        pytest.skip("MSDA CUDA extension is not available.")

    device = torch.device("cuda")
    bsz, n_query, n_heads, head_dim = 2, 24, 4, 8
    dim = n_heads * head_dim

    value, spatial_shapes, level_start_index, sampling_locations, attention_weights = make_msda_case(
        device=device,
        dtype=torch.float32,
        batch_size=bsz,
        n_query=n_query,
        n_heads=n_heads,
        head_dim=head_dim,
        spatial_shapes_hw=[(10, 12), (5, 6)],
        n_points=4,
        requires_grad=False,
    )
    value_flat = value.reshape(bsz, value.shape[1], dim)
    query = torch.randn(bsz, n_query, dim, device=device)
    target = torch.randn(bsz, n_query, dim, device=device)

    dense = _DenseCrossAttnBaseline(dim=dim, n_heads=n_heads).to(device)
    proj = nn.Linear(dim, dim).to(device)
    opt_dense = torch.optim.Adam(list(dense.parameters()) + list(proj.parameters()), lr=1e-3)

    # MSDA branch has learnable query/value projections only (sampling is fixed in this toy setup).
    msda_q = nn.Linear(dim, dim).to(device)
    msda_o = nn.Linear(dim, dim).to(device)
    opt_msda = torch.optim.Adam(list(msda_q.parameters()) + list(msda_o.parameters()), lr=1e-3)

    for _ in range(20):
        opt_dense.zero_grad(set_to_none=True)
        pred = proj(dense(query, value_flat))
        loss = (pred - target).square().mean()
        loss.backward()
        opt_dense.step()

    for _ in range(20):
        opt_msda.zero_grad(set_to_none=True)
        q = msda_q(query)
        out = multi_scale_deformable_attention(
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            sampling_locations=sampling_locations,
            attention_weights=attention_weights,
            prefer_cuda=True,
        )
        pred = msda_o(out.reshape(bsz, n_query, dim) + q)
        loss = (pred - target).square().mean()
        loss.backward()
        opt_msda.step()

    dense_final = (proj(dense(query, value_flat)) - target).square().mean().item()
    msda_final = (
        msda_o(
            multi_scale_deformable_attention(
                value=value,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                sampling_locations=sampling_locations,
                attention_weights=attention_weights,
                prefer_cuda=True,
            ).reshape(bsz, n_query, dim)
            + msda_q(query)
        )
        - target
    ).square().mean().item()

    assert dense_final < 10.0
    assert msda_final < 10.0
