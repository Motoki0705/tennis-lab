"""Boundary and compute tests for the shared sequence discriminator."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.utils.models.architectures import (
    TransformerSequenceDiscriminator,
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


def test_sequence_and_padding_mask_run_public_forward() -> None:
    model = _discriminator().eval()
    sequence = torch.randn(2, 4, 3)
    padding_mask = torch.tensor(
        [[False, False, False, True], [False, False, True, True]]
    )

    with torch.no_grad():
        output = model(sequence, padding_mask=padding_mask)

    assert output.shape == (2,)
    assert torch.isfinite(output).all()


def test_padded_values_are_replaced_before_attention() -> None:
    model = _discriminator().eval()
    sequence = torch.randn(2, 4, 3)
    padding_mask = torch.tensor(
        [[False, True, False, True], [True, True, True, True]]
    )
    changed = sequence.clone()
    changed[padding_mask] = 10_000.0

    with torch.no_grad():
        expected = model(sequence, padding_mask=padding_mask)
        actual = model(changed, padding_mask=padding_mask)

    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()


class _AttentionMaskSpy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention_mask: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        del freqs_cis
        self.attention_mask = attn_mask
        return x


def test_forward_adds_valid_cls_token_and_dense_keep_mask() -> None:
    model = _discriminator().eval()
    spy = _AttentionMaskSpy()
    model.blocks = nn.ModuleList([spy])
    padding_mask = torch.tensor([[False, True, False]])

    with torch.no_grad():
        model(torch.randn(1, 3, 3), padding_mask=padding_mask)

    expected_valid = torch.tensor([[True, True, False, True]])
    expected = expected_valid[:, None, :].expand(1, 4, 4)
    assert torch.equal(spy.attention_mask, expected)


@pytest.mark.parametrize(
    ("sequence", "padding_mask", "error", "message"),
    [
        (
            torch.randn(2, 3),
            torch.zeros(2, 3, dtype=torch.bool),
            ValueError,
            r"shape \(B,T,F\)",
        ),
        (
            torch.randn(2, 4, 2),
            torch.zeros(2, 4, dtype=torch.bool),
            ValueError,
            "feature dimension",
        ),
        (
            torch.randn(2, 7, 3),
            torch.zeros(2, 7, dtype=torch.bool),
            ValueError,
            "max_seq_len",
        ),
        (
            torch.randn(2, 4, 3),
            torch.zeros(2, 4, 1, dtype=torch.bool),
            ValueError,
            r"shape \(B,T\)",
        ),
        (
            torch.randn(2, 4, 3),
            torch.zeros(2, 3, dtype=torch.bool),
            ValueError,
            r"shape \(B,T\)",
        ),
        (
            torch.randn(2, 4, 3),
            torch.zeros(2, 4),
            TypeError,
            "torch.bool",
        ),
    ],
)
def test_forward_rejects_invalid_public_inputs(
    sequence: torch.Tensor,
    padding_mask: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        _discriminator()(sequence, padding_mask=padding_mask)


def test_forward_rejects_padding_mask_on_different_device() -> None:
    sequence = torch.randn(2, 4, 3)
    padding_mask = torch.zeros(2, 4, dtype=torch.bool, device="meta")

    with pytest.raises(ValueError, match="same device"):
        _discriminator()(sequence, padding_mask=padding_mask)
