import pytest
import torch

from src.utils.models.components.ops import is_moe_cuda_available
from src.utils.models.components.ops.moe import (
    moe_combine,
    moe_dispatch,
    reference_moe_dispatch,
)

pytestmark = pytest.mark.cuda


def _require_moe_cuda_extension() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not is_moe_cuda_available():
        pytest.skip("MoE CUDA extension is not built")


def test_cuda_dispatch_combine_matches_reference_roundtrip() -> None:
    _require_moe_cuda_extension()
    tokens = torch.randn(8, 6, device="cuda", dtype=torch.float32, requires_grad=True)
    expert_indices = torch.tensor(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [0, 2],
            [1, 3],
            [2, 0],
            [3, 1],
        ],
        device="cuda",
    )
    expert_weights = torch.full((8, 2), 0.5, device="cuda", requires_grad=True)

    dispatch_result = moe_dispatch(
        tokens,
        expert_indices,
        expert_weights,
        num_experts=4,
        use_cuda=True,
    )
    output = moe_combine(dispatch_result.expert_inputs, dispatch_result, use_cuda=True)

    torch.testing.assert_close(output, tokens)
    output.sum().backward()
    torch.testing.assert_close(tokens.grad, torch.ones_like(tokens))
    assert expert_weights.grad is not None
    assert torch.isfinite(expert_weights.grad).all()


def test_cuda_dispatch_layout_matches_reference_with_capacity_drop() -> None:
    _require_moe_cuda_extension()
    tokens_cpu = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    expert_indices_cpu = torch.tensor(
        [
            [0, 1],
            [0, 1],
            [0, 2],
            [1, 2],
            [2, 0],
            [2, 1],
        ],
        dtype=torch.long,
    )
    expert_weights_cpu = torch.tensor(
        [
            [0.7, 0.3],
            [0.6, 0.4],
            [0.5, 0.5],
            [0.2, 0.8],
            [0.1, 0.9],
            [0.8, 0.2],
        ],
        dtype=torch.float32,
    )

    reference_result = reference_moe_dispatch(
        tokens_cpu,
        expert_indices_cpu,
        expert_weights_cpu,
        num_experts=3,
        capacity=2,
        drop_policy="capacity",
    )
    cuda_result = moe_dispatch(
        tokens_cpu.cuda(),
        expert_indices_cpu.cuda(),
        expert_weights_cpu.cuda(),
        num_experts=3,
        capacity=2,
        drop_policy="capacity",
        use_cuda=True,
    )

    torch.testing.assert_close(
        cuda_result.expert_inputs.cpu(), reference_result.expert_inputs
    )
    torch.testing.assert_close(cuda_result.locations.cpu(), reference_result.locations)
    torch.testing.assert_close(
        cuda_result.combine_mask.cpu(), reference_result.combine_mask
    )
    torch.testing.assert_close(
        cuda_result.expert_counts.cpu(), reference_result.expert_counts
    )
