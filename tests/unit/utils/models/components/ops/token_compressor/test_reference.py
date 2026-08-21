"""CPU oracle tests for token-compressor pooling."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from src.utils.models.components.ops.token_compressor import (
    reference_token_compressor_pool,
)


def _loop_pool(
    raw_kv: Tensor,
    raw_gate: Tensor,
    state_valid: Tensor,
    ratio: int,
) -> tuple[Tensor, Tensor]:
    n, sequence_length, _, head_dim = raw_kv.shape
    accumulation_dtype = (
        torch.float64
        if raw_kv.dtype == torch.float64 or raw_gate.dtype == torch.float64
        else torch.float32
    )
    rows: list[Tensor] = []
    valid_rows: list[Tensor] = []
    for compressed_index in range((sequence_length + ratio - 1) // ratio):
        batch_rows: list[Tensor] = []
        batch_valid: list[Tensor] = []
        for batch_index in range(n):
            values: list[Tensor] = []
            logits: list[Tensor] = []
            for branch, block in ((0, compressed_index - 1), (1, compressed_index)):
                for token in range(block * ratio, (block + 1) * ratio):
                    if 0 <= token < sequence_length and state_valid[batch_index, token]:
                        values.append(raw_kv[batch_index, token, branch])
                        logits.append(raw_gate[batch_index, token, branch])
            if values:
                stacked_values = torch.stack(values).to(accumulation_dtype)
                weights = torch.softmax(torch.stack(logits).to(accumulation_dtype), 0)
                batch_rows.append((weights * stacked_values).sum(0))
                batch_valid.append(torch.tensor(True))
            else:
                batch_rows.append(torch.zeros(head_dim, dtype=accumulation_dtype))
                batch_valid.append(torch.tensor(False))
        rows.append(torch.stack(batch_rows))
        valid_rows.append(torch.stack(batch_valid))
    return torch.stack(rows, 1), torch.stack(valid_rows, 1)


@pytest.mark.parametrize("sequence_length", [1, 3, 4, 5, 17])
def test_reference_matches_direct_previous_current_oracle(
    sequence_length: int,
) -> None:
    generator = torch.Generator().manual_seed(753 + sequence_length)
    raw_kv = torch.randn(3, sequence_length, 2, 7, generator=generator)
    raw_gate = torch.randn(3, sequence_length, 2, 7, generator=generator) * 4
    state_valid = torch.rand(3, sequence_length, generator=generator) > 0.35
    state_valid[1] = False

    expected = _loop_pool(raw_kv, raw_gate, state_valid, 4)
    actual = reference_token_compressor_pool(
        raw_kv, raw_gate, state_valid, compression_ratio=4
    )

    torch.testing.assert_close(actual[0], expected[0], atol=1.0e-6, rtol=1.0e-5)
    torch.testing.assert_close(actual[1], expected[1])
    assert actual[0].dtype == torch.float32
    assert torch.count_nonzero(actual[0][1]) == 0


def test_reference_handles_noncontiguous_mask_invalid_nan_and_large_logits() -> None:
    raw_kv = torch.randn(2, 5, 2, 6)
    raw_gate = torch.zeros_like(raw_kv)
    mask_storage = torch.ones(2, 10, dtype=torch.bool)
    state_valid = mask_storage[:, ::2]
    state_valid[0, 1::2] = False
    state_valid[1] = False
    raw_kv[~state_valid] = torch.nan
    raw_gate[~state_valid] = torch.nan
    raw_gate[0, 0, 1] = 1.0e30
    raw_gate[0, 2, 1] = -1.0e30
    assert not state_valid.is_contiguous()

    pooled, pooled_valid = reference_token_compressor_pool(
        raw_kv, raw_gate, state_valid, compression_ratio=4
    )

    assert torch.isfinite(pooled).all()
    assert torch.count_nonzero(pooled[1]) == 0
    assert not pooled_valid[1].any()
    torch.testing.assert_close(pooled[0, 0], raw_kv[0, 0, 1])


def test_reference_backward_has_native_kv_and_gate_gradient_dtypes() -> None:
    raw_kv = torch.randn(2, 5, 2, 4, dtype=torch.bfloat16, requires_grad=True)
    raw_gate = torch.randn(2, 5, 2, 4, dtype=torch.float32, requires_grad=True)
    state_valid = torch.tensor(
        [[True, False, True, True, True], [False, False, False, False, False]]
    )

    pooled, pooled_valid = reference_token_compressor_pool(
        raw_kv, raw_gate, state_valid, compression_ratio=4
    )
    pooled.square().sum().backward()

    assert pooled.dtype == torch.float32
    assert not pooled_valid.requires_grad
    assert raw_kv.grad is not None and raw_kv.grad.dtype == torch.bfloat16
    assert raw_gate.grad is not None and raw_gate.grad.dtype == torch.float32
    assert torch.count_nonzero(raw_kv.grad[~state_valid]) == 0
    assert torch.count_nonzero(raw_gate.grad[~state_valid]) == 0


@pytest.mark.parametrize(
    ("mutate", "error_type", "message"),
    [
        (lambda kv, gate, mask: (kv[:, :, 0], gate, mask), ValueError, "shape"),
        (lambda kv, gate, mask: (kv, gate[..., :3], mask), ValueError, "shape"),
        (
            lambda kv, gate, mask: (kv, gate, mask.float()),
            TypeError,
            "dtype bool",
        ),
    ],
)
def test_reference_rejects_invalid_runtime_contracts(
    mutate: object, error_type: type[Exception], message: str
) -> None:
    raw_kv = torch.randn(2, 5, 2, 4)
    raw_gate = torch.randn_like(raw_kv)
    state_valid = torch.ones(2, 5, dtype=torch.bool)
    transform = mutate
    assert callable(transform)
    changed = transform(raw_kv, raw_gate, state_valid)
    with pytest.raises(error_type, match=message):
        reference_token_compressor_pool(
            *changed,
            compression_ratio=4,  # type: ignore[arg-type]
        )
