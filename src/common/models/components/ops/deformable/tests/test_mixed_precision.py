"""Mixed precision stability tests for deformable attention."""

from __future__ import annotations

import pytest
import torch

from src.common.models.components.ops.deformable.api import multi_scale_deformable_attention
from src.common.models.components.ops.deformable.kernels.msda_dispatch import get_msda_extension
from src.common.models.components.ops.deformable.tests._fixtures import make_msda_case


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_msda_mixed_precision_stability(dtype: torch.dtype) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is not supported on this GPU.")
    ext = get_msda_extension()
    if ext is None:
        pytest.skip("MSDA CUDA extension is not available.")

    device = torch.device("cuda")
    value, spatial_shapes, level_start_index, sampling_locations, attention_weights = make_msda_case(
        device=device,
        dtype=dtype,
        batch_size=2,
        n_query=48,
        n_heads=8,
        head_dim=8,
        spatial_shapes_hw=[(20, 32), (10, 16)],
        n_points=4,
        requires_grad=True,
    )

    optimizer = torch.optim.Adam([value, sampling_locations, attention_weights], lr=1e-3)

    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        out = multi_scale_deformable_attention(
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            sampling_locations=sampling_locations,
            attention_weights=attention_weights,
            prefer_cuda=True,
        )
        loss = out.square().mean()
        assert torch.isfinite(loss).item()
        loss.backward()

        assert value.grad is not None and torch.isfinite(value.grad).all().item()
        assert sampling_locations.grad is not None and torch.isfinite(sampling_locations.grad).all().item()
        assert attention_weights.grad is not None and torch.isfinite(attention_weights.grad).all().item()
        optimizer.step()
