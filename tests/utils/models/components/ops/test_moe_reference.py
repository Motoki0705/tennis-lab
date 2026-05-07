import torch

from src.utils.models.components.ops.moe import moe_combine, moe_dispatch


def test_reference_dispatch_combine_roundtrip() -> None:
    tokens = torch.arange(24, dtype=torch.float32).reshape(6, 4).requires_grad_(True)
    expert_indices = torch.tensor(
        [
            [0, 1],
            [1, 2],
            [2, 0],
            [0, 2],
            [1, 0],
            [2, 1],
        ]
    )
    expert_weights = torch.tensor(
        [
            [0.75, 0.25],
            [0.5, 0.5],
            [0.2, 0.8],
            [1.0, 0.0],
            [0.4, 0.6],
            [0.1, 0.9],
        ],
        dtype=tokens.dtype,
    )

    dispatch_result = moe_dispatch(
        tokens,
        expert_indices,
        expert_weights,
        num_experts=3,
        use_cuda=False,
    )
    output = moe_combine(dispatch_result.expert_inputs, dispatch_result, use_cuda=False)

    expected = tokens * expert_weights.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    torch.testing.assert_close(tokens.grad, torch.ones_like(tokens))


def test_reference_capacity_drop_masks_overflow_assignments() -> None:
    tokens = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    expert_indices = torch.zeros(4, 1, dtype=torch.long)
    expert_weights = torch.ones(4, 1)

    dispatch_result = moe_dispatch(
        tokens,
        expert_indices,
        expert_weights,
        num_experts=2,
        capacity=2,
        drop_policy="capacity",
        use_cuda=False,
    )
    output = moe_combine(dispatch_result.expert_inputs, dispatch_result, use_cuda=False)

    assert dispatch_result.capacity == 2
    assert dispatch_result.combine_mask.sum().item() == 2
    torch.testing.assert_close(output[:2], tokens[:2])
    torch.testing.assert_close(output[2:], torch.zeros_like(tokens[2:]))
