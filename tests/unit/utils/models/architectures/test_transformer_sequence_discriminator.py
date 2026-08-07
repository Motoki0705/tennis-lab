"""Boundary and compute tests for the shared sequence discriminator."""

from __future__ import annotations

import pytest
import torch

from src.utils.models.architectures import (
    TransformerSequenceDiscriminator,
    prepare_sequence_discriminator_inputs,
)


def _discriminator() -> TransformerSequenceDiscriminator:
    return TransformerSequenceDiscriminator(
        input_dim=3,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        ffn_dim=16,
        dropout=0.0,
        rope_dim=4,
        rope_theta=10_000.0,
        ffn_type="mlp",
        max_seq_len=6,
        invalid_init_std=0.02,
        cls_init_std=0.02,
    )


def test_validated_sequence_runs_tensor_only_forward() -> None:
    model = _discriminator().eval()
    sequence = torch.randn(2, 4, 3)
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])

    inputs = prepare_sequence_discriminator_inputs(model, sequence, mask=mask)
    with torch.no_grad():
        output = model(
            inputs.sequence,
            token_mask=inputs.token_mask,
            attention_mask=inputs.attention_mask,
        )

    assert output.shape == (2,)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    ("sequence", "mask", "message"),
    [
        (torch.randn(2, 3), torch.ones(2, 3, dtype=torch.bool), "must be \(B, T, F\)"),
        (torch.randn(2, 4, 2), torch.ones(2, 4, dtype=torch.bool), "feature dimension"),
        (torch.randn(2, 7, 3), torch.ones(2, 7, dtype=torch.bool), "max_seq_len"),
        (torch.randn(2, 4, 3), torch.ones(2, 3), "shape \(B, T\)"),
        (torch.randn(2, 4, 3), torch.ones(2, 4), "torch.bool"),
    ],
)
def test_boundary_rejects_invalid_sequence_before_forward(
    sequence: torch.Tensor,
    mask: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        prepare_sequence_discriminator_inputs(
            _discriminator(),
            sequence,
            mask=mask,
        )
