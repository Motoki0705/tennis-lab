"""Oracle and contract tests for gather-based compressed attention."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from src.utils.models.components.ops.compressed_time_local import (
    api as compressed_api,
)
from src.utils.models.components.ops.compressed_time_local import (
    reference as compressed_reference,
)
from src.utils.models.components.ops.compressed_time_local.api import (
    resolve_compressed_time_local_attention,
)
from src.utils.models.components.ops.compressed_time_local.reference import (
    reference_compressed_time_local_attention,
)


def _dense_oracle(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    query_valid: Tensor,
    key_valid: Tensor,
    compression_ratio: int,
    window_radius: int,
) -> Tensor:
    """Explicit dense ``[T,Tc]`` oracle used only by tests."""
    query_len = query.shape[2]
    key_len = key.shape[2]
    query_positions = torch.arange(query_len, device=query.device)
    key_positions = torch.arange(key_len, device=query.device)
    centers = torch.div(query_positions, compression_ratio, rounding_mode="floor")
    local = (key_positions[None, :] - centers[:, None]).abs() <= window_radius
    keep = query_valid[:, :, None] & key_valid[:, None, :] & local[None, :, :]
    if bool((query_valid & ~keep.any(dim=-1)).any()):
        raise RuntimeError("oracle valid row has no key")

    safe_query = torch.where(
        query_valid[:, None, :, None], query, torch.zeros_like(query)
    )
    safe_key = torch.where(key_valid[:, None, :, None], key, torch.zeros_like(key))
    safe_value = torch.where(
        key_valid[:, None, :, None], value, torch.zeros_like(value)
    )
    scores = torch.einsum("nhtd,nhcd->nhtc", safe_query, safe_key) / math.sqrt(
        query.shape[-1]
    )
    safe_keep = keep.clone()
    empty = ~safe_keep.any(dim=-1)
    safe_keep[..., 0] |= empty
    scores = scores.masked_fill(~safe_keep[:, None], -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.einsum("nhtc,nhcd->nhtd", probabilities, safe_value)
    return torch.where(query_valid[:, None, :, None], output, torch.zeros_like(output))


def _inputs(
    *, dtype: torch.dtype = torch.float32
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    torch.manual_seed(7)
    query = torch.randn(2, 3, 7, 4, dtype=dtype)
    key = torch.randn(2, 3, 3, 4, dtype=dtype)
    value = torch.randn(2, 3, 3, 4, dtype=dtype)
    query_valid = torch.tensor(
        [
            [True, True, False, True, True, False, True],
            [False, False, False, False, False, False, False],
        ]
    )
    key_valid = torch.tensor(
        [
            [True, True, True],
            [False, False, False],
        ]
    )
    return query, key, value, query_valid, key_valid


def test_gather_reference_matches_dense_oracle_forward() -> None:
    query, key, value, query_valid, key_valid = _inputs()

    actual = reference_compressed_time_local_attention(
        query,
        key,
        value,
        query_valid=query_valid,
        key_valid=key_valid,
        compression_ratio=3,
        window_radius=1,
    )
    expected = _dense_oracle(
        query,
        key,
        value,
        query_valid=query_valid,
        key_valid=key_valid,
        compression_ratio=3,
        window_radius=1,
    )

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    assert torch.count_nonzero(actual[1]) == 0
    assert torch.count_nonzero(actual[0, :, ~query_valid[0]]) == 0


def test_gather_reference_matches_dense_oracle_backward() -> None:
    query, key, value, query_valid, key_valid = _inputs(dtype=torch.float64)
    actual_inputs = tuple(
        tensor.clone().requires_grad_() for tensor in (query, key, value)
    )
    oracle_inputs = tuple(
        tensor.clone().requires_grad_() for tensor in (query, key, value)
    )
    upstream = torch.randn_like(query)

    actual = reference_compressed_time_local_attention(
        *actual_inputs,
        query_valid=query_valid,
        key_valid=key_valid,
        compression_ratio=3,
        window_radius=1,
    )
    expected = _dense_oracle(
        *oracle_inputs,
        query_valid=query_valid,
        key_valid=key_valid,
        compression_ratio=3,
        window_radius=1,
    )
    actual_gradients = torch.autograd.grad((actual * upstream).sum(), actual_inputs)
    expected_gradients = torch.autograd.grad((expected * upstream).sum(), oracle_inputs)

    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, atol=1e-12, rtol=1e-10
        )


def test_boundary_windows_select_first_middle_and_last_compressed_states() -> None:
    query = torch.zeros(1, 1, 9, 2)
    key = torch.zeros(1, 1, 3, 2)
    value = torch.tensor([[[[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]]])
    query_valid = torch.ones(1, 9, dtype=torch.bool)
    key_valid = torch.ones(1, 3, dtype=torch.bool)

    output = reference_compressed_time_local_attention(
        query,
        key,
        value,
        query_valid=query_valid,
        key_valid=key_valid,
        compression_ratio=4,
        window_radius=1,
    )

    torch.testing.assert_close(output[0, 0, 0], torch.tensor([15.0, 0.0]))
    torch.testing.assert_close(output[0, 0, 4], torch.tensor([20.0, 0.0]))
    torch.testing.assert_close(output[0, 0, 8], torch.tensor([25.0, 0.0]))


def test_invalid_padding_values_do_not_affect_valid_outputs() -> None:
    query, key, value, query_valid, key_valid = _inputs()
    expected = reference_compressed_time_local_attention(
        query,
        key,
        value,
        query_valid=query_valid,
        key_valid=key_valid,
        compression_ratio=3,
        window_radius=1,
    )
    changed_query = query.clone()
    changed_key = key.clone()
    changed_value = value.clone()
    changed_query[~query_valid[:, None, :, None].expand_as(query)] = torch.nan
    changed_key[~key_valid[:, None, :, None].expand_as(key)] = torch.nan
    changed_value[~key_valid[:, None, :, None].expand_as(value)] = torch.nan

    actual = reference_compressed_time_local_attention(
        changed_query,
        changed_key,
        changed_value,
        query_valid=query_valid,
        key_valid=key_valid,
        compression_ratio=3,
        window_radius=1,
    )

    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()


def test_valid_query_without_local_key_is_rejected() -> None:
    query = torch.randn(1, 1, 4, 2)
    key = torch.randn(1, 1, 2, 2)
    value = torch.randn(1, 1, 2, 2)
    query_valid = torch.tensor([[True, False, False, False]])
    key_valid = torch.tensor([[False, True]])

    with pytest.raises(RuntimeError, match="no valid compressed key"):
        reference_compressed_time_local_attention(
            query,
            key,
            value,
            query_valid=query_valid,
            key_valid=key_valid,
            compression_ratio=2,
            window_radius=0,
        )


def test_all_invalid_sample_returns_exact_zero() -> None:
    query = torch.full((2, 2, 5, 4), torch.nan)
    key = torch.full((2, 2, 3, 4), torch.nan)
    value = torch.full((2, 2, 3, 4), torch.nan)
    output = reference_compressed_time_local_attention(
        query,
        key,
        value,
        query_valid=torch.zeros(2, 5, dtype=torch.bool),
        key_valid=torch.zeros(2, 3, dtype=torch.bool),
        compression_ratio=2,
        window_radius=1,
    )

    assert torch.count_nonzero(output) == 0
    assert torch.isfinite(output).all()


def test_non_contiguous_tensors_match_contiguous_tensors() -> None:
    query_base = torch.randn(2, 3, 7, 8)
    key_base = torch.randn(2, 3, 3, 8)
    value_base = torch.randn(2, 3, 3, 8)
    query = query_base[..., ::2]
    key = key_base[..., ::2]
    value = value_base[..., ::2]
    query_valid = torch.ones(2, 14, dtype=torch.bool)[:, ::2]
    key_valid = torch.ones(2, 6, dtype=torch.bool)[:, ::2]
    query_valid[0, 2] = False
    assert not query.is_contiguous()
    assert not key.is_contiguous()
    assert not value.is_contiguous()
    assert not query_valid.is_contiguous()

    actual = reference_compressed_time_local_attention(
        query,
        key,
        value,
        query_valid=query_valid,
        key_valid=key_valid,
        compression_ratio=3,
        window_radius=1,
    )
    expected = reference_compressed_time_local_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        query_valid=query_valid.contiguous(),
        key_valid=key_valid.contiguous(),
        compression_ratio=3,
        window_radius=1,
    )

    torch.testing.assert_close(actual, expected)


def test_small_double_gradcheck() -> None:
    query = torch.randn(1, 1, 3, 2, dtype=torch.float64, requires_grad=True)
    key = torch.randn(1, 1, 2, 2, dtype=torch.float64, requires_grad=True)
    value = torch.randn(1, 1, 2, 2, dtype=torch.float64, requires_grad=True)
    query_valid = torch.tensor([[True, False, True]])
    key_valid = torch.tensor([[True, True]])

    def attend(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return reference_compressed_time_local_attention(
            q,
            k,
            v,
            query_valid=query_valid,
            key_valid=key_valid,
            compression_ratio=2,
            window_radius=1,
        )

    assert torch.autograd.gradcheck(
        attend,
        (query, key, value),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )


def test_dropout_off_is_deterministic_and_training_dropout_is_finite() -> None:
    query, key, value, query_valid, key_valid = _inputs()

    def run(training: bool) -> Tensor:
        return reference_compressed_time_local_attention(
            query,
            key,
            value,
            query_valid=query_valid,
            key_valid=key_valid,
            compression_ratio=3,
            window_radius=1,
            dropout_p=0.5,
            training=training,
        )

    torch.manual_seed(1)
    eval_first = run(training=False)
    torch.manual_seed(99)
    eval_second = run(training=False)
    torch.manual_seed(23)
    train_first = run(training=True)
    torch.manual_seed(23)
    train_second = run(training=True)

    torch.testing.assert_close(eval_first, eval_second, rtol=0, atol=0)
    torch.testing.assert_close(train_first, train_second, rtol=0, atol=0)
    assert train_first.shape == query.shape
    assert torch.isfinite(train_first).all()


def test_sdpa_receives_fixed_window_not_dense_query_by_key_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = torch.randn(1, 2, 33, 4)
    key = torch.randn(1, 2, 9, 4)
    value = torch.randn(1, 2, 9, 4)
    query_valid = torch.ones(1, 33, dtype=torch.bool)
    key_valid = torch.ones(1, 9, dtype=torch.bool)
    original_sdpa = compressed_reference.F.scaled_dot_product_attention
    observed_shapes: list[tuple[torch.Size, torch.Size, torch.Size]] = []

    def inspect_sdpa(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        attn_mask: Tensor,
        dropout_p: float,
        is_causal: bool,
    ) -> Tensor:
        observed_shapes.append((q.shape, k.shape, attn_mask.shape))
        return original_sdpa(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

    monkeypatch.setattr(
        compressed_reference.F, "scaled_dot_product_attention", inspect_sdpa
    )
    reference_compressed_time_local_attention(
        query,
        key,
        value,
        query_valid=query_valid,
        key_valid=key_valid,
        compression_ratio=4,
        window_radius=2,
    )

    assert observed_shapes == [
        (
            torch.Size([1, 2, 33, 1, 4]),
            torch.Size([1, 2, 33, 5, 4]),
            torch.Size([1, 1, 33, 1, 5]),
        )
    ]
    assert all(
        shape[-2:] != torch.Size([33, 9])
        for shapes in observed_shapes
        for shape in shapes
    )


def test_backend_resolution_binds_reference_policy() -> None:
    executor = resolve_compressed_time_local_attention(
        "reference", compression_ratio=2, window_radius=0
    )
    query = torch.randn(1, 1, 3, 2)
    key = torch.randn(1, 1, 2, 2)
    output = executor(
        query,
        key,
        key,
        query_valid=torch.ones(1, 3, dtype=torch.bool),
        key_valid=torch.ones(1, 2, dtype=torch.bool),
    )

    assert output.shape == query.shape


def test_cuda_backend_fails_at_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable() -> Callable[..., Tensor]:
        raise RuntimeError("extension unavailable at construction")

    monkeypatch.setattr(compressed_api, "_require_cuda_executor", unavailable)
    with pytest.raises(RuntimeError, match="at construction"):
        resolve_compressed_time_local_attention(
            "cuda", compression_ratio=2, window_radius=1
        )


@pytest.mark.parametrize(
    ("backend", "ratio", "radius", "message"),
    [
        ("automatic", 2, 1, "Unsupported"),
        ("reference", 1, 1, "at least 2"),
        ("reference", 2, -1, "non-negative int"),
    ],
)
def test_backend_resolution_rejects_invalid_policy(
    backend: str,
    ratio: int,
    radius: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_compressed_time_local_attention(
            backend,  # type: ignore[arg-type]
            compression_ratio=ratio,
            window_radius=radius,
        )
