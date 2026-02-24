"""Scale stress tests for deformable attention (OOM/latency gates)."""

from __future__ import annotations

import pytest
import torch

from src.utils.models.components.ops.deformable.api import multi_scale_deformable_attention
from src.utils.models.components.ops.deformable.kernels.msda_dispatch import get_msda_extension
from src.utils.models.components.ops.deformable.tests._fixtures import cuda_peak_mem_mb, cuda_step_time_ms, env_float, make_msda_case


@pytest.mark.cuda
@pytest.mark.slow
def test_msda_scale_stress_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    ext = get_msda_extension()
    if ext is None:
        pytest.skip("MSDA CUDA extension is not available.")

    max_ms = env_float("MSDA_STRESS_MAX_MS", 80.0)
    max_mem_mb = env_float("MSDA_STRESS_MAX_MEM_MB", 4096.0)

    device = torch.device("cuda")
    value, spatial_shapes, level_start_index, sampling_locations, attention_weights = make_msda_case(
        device=device,
        dtype=torch.float32,
        batch_size=2,
        n_query=192,
        n_heads=8,
        head_dim=16,
        spatial_shapes_hw=[(36, 64), (18, 32)],
        n_points=4,
        requires_grad=False,
    )

    def forward() -> torch.Tensor:
        return multi_scale_deformable_attention(
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            sampling_locations=sampling_locations,
            attention_weights=attention_weights,
            prefer_cuda=True,
        )

    peak_mem = cuda_peak_mem_mb(forward)
    latency_ms = cuda_step_time_ms(forward, warmup=2, iters=8)
    out = forward()

    assert torch.isfinite(out).all().item()
    assert peak_mem <= max_mem_mb
    assert latency_ms <= max_ms
