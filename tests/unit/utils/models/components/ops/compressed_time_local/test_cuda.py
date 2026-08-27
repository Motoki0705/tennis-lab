"""CUDA tests for fused compressed time-local attention."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import textwrap
from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from src.utils.models.components.ops.compressed_time_local import (
    reference_compressed_time_local_attention,
    resolve_compressed_time_local_attention,
)


def _apply_full_rope_explicit(values: Tensor, freqs_cis: Tensor) -> Tensor:
    values_nthd = values.transpose(1, 2)
    values_complex = torch.view_as_complex(
        values_nthd.float().reshape(*values_nthd.shape[:-1], -1, 2)
    )
    rotated = torch.view_as_real(values_complex * freqs_cis.to(torch.complex64))
    return rotated.flatten(-2).to(values.dtype).transpose(1, 2)


def _phasors(
    *,
    batch_size: int,
    sequence_length: int,
    heads: int,
    pairs: int,
    rank: int,
    device: torch.device,
) -> Tensor:
    if rank == 3:
        angles = torch.randn(sequence_length, 1, pairs * 2, device=device)
        return torch.polar(torch.ones_like(angles), angles)[..., ::2]
    angle_storage = torch.randn(
        batch_size,
        heads,
        sequence_length,
        pairs * 2,
        device=device,
    )
    return torch.polar(torch.ones_like(angle_storage), angle_storage).transpose(1, 2)[
        ..., ::2
    ]


pytestmark = pytest.mark.skipif(
    os.environ.get("TENNIS_LAB_RUN_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
    reason="CUDA operation tests require TENNIS_LAB_RUN_CUDA_TESTS=1 and CUDA",
)


def _is_alias_of(left: Tensor, right: Tensor) -> bool:
    """Call PyTorch's runtime alias probe, which is absent from its type stubs."""
    return bool(torch._C._is_alias_of(left, right))  # type: ignore[attr-defined]


def _inputs(
    dtype: torch.dtype,
    *,
    key_heads: int = 3,
    query_layout: str = "contiguous",
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(753)
    if query_layout == "contiguous":
        query = torch.randn(
            2, 3, 37, 64, device="cuda", dtype=dtype, generator=generator
        )
    elif query_layout == "production":
        query = torch.randn(
            2, 37, 3, 64, device="cuda", dtype=dtype, generator=generator
        ).transpose(1, 2)
    else:
        raise ValueError(f"unknown query layout: {query_layout}")
    key = torch.randn(
        2, key_heads, 10, 128, device="cuda", dtype=dtype, generator=generator
    )[..., ::2]
    value = torch.randn(
        2, key_heads, 10, 128, device="cuda", dtype=dtype, generator=generator
    )[..., ::2]
    query_valid = torch.tensor(
        [[index % 4 != 1 for index in range(37)], [False] * 37],
        dtype=torch.bool,
        device="cuda",
    )
    key_valid = torch.tensor(
        [[True, False, True, True, False, True, True, False, True, True], [False] * 10],
        dtype=torch.bool,
        device="cuda",
    )
    return query, key, value, query_valid, key_valid


def _shared_inputs(
    dtype: torch.dtype,
    *,
    query_layout: str = "contiguous",
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    query, key, value, query_valid, key_valid = _inputs(
        dtype,
        key_heads=1,
        query_layout=query_layout,
    )
    query_valid[1] = torch.tensor(
        [index % 5 != 2 for index in range(37)], dtype=torch.bool, device="cuda"
    )
    key_valid[1] = True
    return query, key, value, query_valid, key_valid


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.float32, 2.0e-5, 2.0e-4),
        (torch.float16, 6.5e-2, 2.0e-2),
        (torch.bfloat16, 6.5e-2, 2.0e-2),
    ],
)
@pytest.mark.parametrize("query_layout", ["contiguous", "production"])
def test_cuda_matches_reference_forward_and_all_qkv_gradients(
    dtype: torch.dtype,
    atol: float,
    rtol: float,
    query_layout: str,
) -> None:
    inputs = _inputs(dtype, query_layout=query_layout)
    reference_inputs = [tensor.detach().clone() for tensor in inputs[:3]]
    cuda_inputs = [tensor.detach().clone() for tensor in inputs[:3]]
    for tensor in (*reference_inputs, *cuda_inputs):
        tensor.requires_grad_(True)
    query_valid, key_valid = inputs[3:]
    reference = reference_compressed_time_local_attention(
        *reference_inputs,
        query_valid=query_valid,
        key_valid=key_valid,
        compression_ratio=4,
        window_radius=2,
    )
    executor = resolve_compressed_time_local_attention(
        "cuda", compression_ratio=4, window_radius=2
    )
    actual = executor(
        *cuda_inputs,
        query_valid=query_valid,
        key_valid=key_valid,
    )
    upstream = torch.randn_like(cuda_inputs[0])
    reference.backward(upstream)
    actual.backward(upstream)

    torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
    assert actual.stride() == (
        actual.shape[1] * actual.shape[2] * actual.shape[3],
        actual.shape[3],
        actual.shape[1] * actual.shape[3],
        1,
    )
    assert not _is_alias_of(actual, cuda_inputs[0])
    round_trip = actual.transpose(1, 2)
    flattened = round_trip.reshape(2, 37, 3 * 64)
    assert round_trip.is_contiguous()
    assert flattened.untyped_storage().data_ptr() == actual.untyped_storage().data_ptr()
    for actual_input, reference_input in zip(
        cuda_inputs, reference_inputs, strict=True
    ):
        torch.testing.assert_close(
            actual_input.grad,
            reference_input.grad,
            atol=atol,
            rtol=rtol,
        )
    assert cuda_inputs[0].grad is not None
    assert (
        torch.count_nonzero(
            cuda_inputs[0].grad.masked_select(~query_valid[:, None, :, None])
        )
        == 0
    )
    for state in cuda_inputs[1:]:
        assert state.grad is not None
        assert (
            torch.count_nonzero(state.grad.masked_select(~key_valid[:, None, :, None]))
            == 0
        )
    assert torch.count_nonzero(actual[1]) == 0


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.float32, 4.0e-5, 4.0e-4),
        (torch.float16, 8.0e-2, 3.0e-2),
        (torch.bfloat16, 8.0e-2, 3.0e-2),
    ],
)
@pytest.mark.parametrize("query_layout", ["contiguous", "production"])
def test_cuda_shared_kv_matches_reference_forward_and_all_qkv_gradients(
    dtype: torch.dtype,
    atol: float,
    rtol: float,
    query_layout: str,
) -> None:
    reference_data = _shared_inputs(dtype, query_layout=query_layout)
    cuda_data = _shared_inputs(dtype, query_layout=query_layout)
    reference_inputs = [tensor.requires_grad_(True) for tensor in reference_data[:3]]
    cuda_inputs = [tensor.requires_grad_(True) for tensor in cuda_data[:3]]
    query_valid, key_valid = reference_data[3:]
    reference = reference_compressed_time_local_attention(
        *reference_inputs,
        query_valid=query_valid,
        key_valid=key_valid,
        compression_ratio=4,
        window_radius=2,
    )
    executor = resolve_compressed_time_local_attention(
        "cuda", compression_ratio=4, window_radius=2
    )
    actual = executor(
        *cuda_inputs,
        query_valid=query_valid,
        key_valid=key_valid,
    )
    upstream = torch.randn_like(reference)
    reference.backward(upstream)
    actual.backward(upstream)

    torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
    for actual_input, reference_input in zip(
        cuda_inputs, reference_inputs, strict=True
    ):
        assert actual_input.grad is not None
        assert reference_input.grad is not None
        assert actual_input.grad.shape == actual_input.shape
        torch.testing.assert_close(
            actual_input.grad,
            reference_input.grad,
            atol=atol,
            rtol=rtol,
        )
    invalid_output = actual.masked_select(~query_valid[:, None, :, None])
    assert torch.count_nonzero(invalid_output) == 0


def test_cuda_shared_tied_key_value_accumulates_both_gradients() -> None:
    reference_data = _shared_inputs(torch.float32)
    cuda_data = _shared_inputs(torch.float32)
    reference_query = reference_data[0].requires_grad_(True)
    reference_state = reference_data[1].requires_grad_(True)
    cuda_query = cuda_data[0].requires_grad_(True)
    cuda_state = cuda_data[1].requires_grad_(True)
    query_valid, key_valid = reference_data[3:]
    reference = reference_compressed_time_local_attention(
        reference_query,
        reference_state,
        reference_state,
        query_valid=query_valid,
        key_valid=key_valid,
        compression_ratio=4,
        window_radius=2,
    )
    executor = resolve_compressed_time_local_attention(
        "cuda", compression_ratio=4, window_radius=2
    )
    actual = executor(
        cuda_query,
        cuda_state,
        cuda_state,
        query_valid=query_valid,
        key_valid=key_valid,
    )
    upstream = torch.randn_like(reference)
    reference.backward(upstream)
    actual.backward(upstream)

    torch.testing.assert_close(actual, reference, atol=4.0e-5, rtol=4.0e-4)
    assert cuda_query.grad is not None
    assert reference_query.grad is not None
    assert cuda_state.grad is not None
    assert reference_state.grad is not None
    assert cuda_state.grad.shape == cuda_state.shape
    torch.testing.assert_close(
        cuda_query.grad, reference_query.grad, atol=4.0e-5, rtol=4.0e-4
    )
    torch.testing.assert_close(
        cuda_state.grad, reference_state.grad, atol=4.0e-5, rtol=4.0e-4
    )


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.float32, 4.0e-5, 4.0e-4),
        (torch.float16, 8.0e-2, 3.0e-2),
        (torch.bfloat16, 8.0e-2, 3.0e-2),
    ],
)
@pytest.mark.parametrize("head_dim", [16, 32, 64])
def test_cuda_fused_rope_subwarp_widths_match_tied_reference(
    dtype: torch.dtype,
    atol: float,
    rtol: float,
    head_dim: int,
) -> None:
    """Exercise packed sub-warp paths, including their partial final warp."""
    torch.manual_seed(753)
    batch_size, heads, query_length = 1, 3, 17
    key_length = 5
    reference_query = torch.randn(
        batch_size,
        query_length,
        heads,
        head_dim,
        device="cuda",
        dtype=dtype,
    ).transpose(1, 2)
    actual_query = reference_query.detach().clone()
    reference_key = torch.randn(
        batch_size,
        1,
        key_length,
        head_dim,
        device="cuda",
        dtype=dtype,
    )
    actual_key = reference_key.detach().clone()
    reference_query.requires_grad_(True)
    actual_query.requires_grad_(True)
    reference_key.requires_grad_(True)
    actual_key.requires_grad_(True)
    query_valid = torch.tensor(
        [[index % 5 != 2 for index in range(query_length)]],
        device="cuda",
        dtype=torch.bool,
    )
    key_valid = torch.tensor(
        [[True, False, True, True, True]],
        device="cuda",
        dtype=torch.bool,
    )
    query_freqs_cis = _phasors(
        batch_size=batch_size,
        sequence_length=query_length,
        heads=heads,
        pairs=head_dim // 2,
        rank=3,
        device=actual_query.device,
    )
    key_freqs_cis = _phasors(
        batch_size=batch_size,
        sequence_length=key_length,
        heads=1,
        pairs=head_dim // 2,
        rank=3,
        device=actual_query.device,
    )
    expected = reference_compressed_time_local_attention(
        _apply_full_rope_explicit(reference_query, query_freqs_cis),
        _apply_full_rope_explicit(reference_key, key_freqs_cis),
        reference_key,
        query_valid=query_valid,
        key_valid=key_valid,
        compression_ratio=4,
        window_radius=2,
    )
    executor = resolve_compressed_time_local_attention(
        "cuda", compression_ratio=4, window_radius=2
    )
    actual = executor(
        actual_query,
        actual_key,
        actual_key,
        query_valid=query_valid,
        key_valid=key_valid,
        query_freqs_cis=query_freqs_cis,
        key_freqs_cis=key_freqs_cis,
    )
    upstream = torch.randn_like(actual)
    expected_gradients = torch.autograd.grad(
        (expected * upstream).sum(),
        (reference_query, reference_key),
    )
    actual_gradients = torch.autograd.grad(
        (actual * upstream).sum(),
        (actual_query, actual_key),
    )

    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            atol=atol,
            rtol=rtol,
        )


@pytest.mark.parametrize(
    ("dtype", "key_heads", "phasor_rank", "tied_kv", "atol", "rtol"),
    [
        (torch.float32, 1, 3, True, 4.0e-5, 4.0e-4),
        (torch.float32, 3, 4, False, 4.0e-5, 4.0e-4),
        (torch.float16, 1, 4, False, 8.0e-2, 3.0e-2),
        (torch.float16, 3, 3, True, 8.0e-2, 3.0e-2),
        (torch.bfloat16, 1, 3, False, 8.0e-2, 3.0e-2),
        (torch.bfloat16, 3, 4, True, 8.0e-2, 3.0e-2),
    ],
)
def test_cuda_fused_rope_matches_explicit_forward_and_raw_qkv_gradients(
    dtype: torch.dtype,
    key_heads: int,
    phasor_rank: int,
    tied_kv: bool,
    atol: float,
    rtol: float,
) -> None:
    torch.manual_seed(753)
    batch_size, heads, query_length, head_dim = 2, 3, 37, 64
    key_length = 10
    packed_reference = torch.randn(
        batch_size,
        query_length,
        heads * head_dim + 80,
        device="cuda",
        dtype=dtype,
    )
    packed_actual = packed_reference.clone()
    reference_query = (
        packed_reference[..., : heads * head_dim]
        .reshape(batch_size, query_length, heads, head_dim)
        .transpose(1, 2)
    )
    actual_query = (
        packed_actual[..., : heads * head_dim]
        .reshape(batch_size, query_length, heads, head_dim)
        .transpose(1, 2)
    )
    reference_key = torch.randn(
        batch_size,
        key_heads,
        key_length,
        head_dim,
        device="cuda",
        dtype=dtype,
    )
    actual_key = reference_key.clone()
    if tied_kv:
        reference_value = reference_key
        actual_value = actual_key
    else:
        reference_value = torch.randn_like(reference_key)
        actual_value = reference_value.clone()
    query_valid = torch.tensor(
        [
            [index % 5 != 2 for index in range(query_length)],
            [False] * query_length,
        ],
        device="cuda",
        dtype=torch.bool,
    )
    key_valid = torch.tensor(
        [[True, False, True, True, False, True, True, False, True, True], [False] * 10],
        device="cuda",
        dtype=torch.bool,
    )
    reference_query.masked_fill_(~query_valid[:, None, :, None], torch.nan)
    actual_query.masked_fill_(~query_valid[:, None, :, None], torch.nan)
    reference_key.masked_fill_(~key_valid[:, None, :, None], torch.nan)
    actual_key.masked_fill_(~key_valid[:, None, :, None], torch.nan)
    reference_value.masked_fill_(~key_valid[:, None, :, None], torch.nan)
    actual_value.masked_fill_(~key_valid[:, None, :, None], torch.nan)
    reference_query.requires_grad_(True)
    actual_query.requires_grad_(True)
    reference_key.requires_grad_(True)
    actual_key.requires_grad_(True)
    if not tied_kv:
        reference_value.requires_grad_(True)
        actual_value.requires_grad_(True)

    query_freqs_cis = _phasors(
        batch_size=batch_size,
        sequence_length=query_length,
        heads=heads,
        pairs=head_dim // 2,
        rank=phasor_rank,
        device=actual_query.device,
    )
    key_freqs_cis = _phasors(
        batch_size=batch_size,
        sequence_length=key_length,
        heads=key_heads,
        pairs=head_dim // 2,
        rank=phasor_rank,
        device=actual_query.device,
    )
    if phasor_rank == 4 and key_heads == 1:
        query_freqs_cis = query_freqs_cis[:1, :, :1]
        key_freqs_cis = key_freqs_cis[:1]
    rotated_query = _apply_full_rope_explicit(reference_query, query_freqs_cis)
    rotated_key = _apply_full_rope_explicit(reference_key, key_freqs_cis)
    expected = reference_compressed_time_local_attention(
        rotated_query,
        rotated_key,
        reference_value,
        query_valid=query_valid,
        key_valid=key_valid,
        compression_ratio=4,
        window_radius=2,
    )
    executor = resolve_compressed_time_local_attention(
        "cuda", compression_ratio=4, window_radius=2
    )
    actual = executor(
        actual_query,
        actual_key,
        actual_value,
        query_valid=query_valid,
        key_valid=key_valid,
        query_freqs_cis=query_freqs_cis,
        key_freqs_cis=key_freqs_cis,
    )
    upstream = torch.randn_like(actual)
    reference_inputs = (
        (reference_query, reference_key)
        if tied_kv
        else (reference_query, reference_key, reference_value)
    )
    actual_inputs = (
        (actual_query, actual_key)
        if tied_kv
        else (actual_query, actual_key, actual_value)
    )
    expected_gradients = torch.autograd.grad(
        (expected * upstream).sum(), reference_inputs
    )
    actual_gradients = torch.autograd.grad((actual * upstream).sum(), actual_inputs)

    assert actual.stride() == (
        query_length * heads * head_dim,
        head_dim,
        heads * head_dim,
        1,
    )
    assert actual.transpose(1, 2).is_contiguous()
    assert actual_query.stride() == (
        query_length * (heads * head_dim + 80),
        head_dim,
        heads * head_dim + 80,
        1,
    )
    if phasor_rank == 4:
        assert not query_freqs_cis.is_contiguous()
        assert not key_freqs_cis.is_contiguous()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        assert actual_gradient.shape == expected_gradient.shape
        torch.testing.assert_close(
            actual_gradient, expected_gradient, atol=atol, rtol=rtol
        )
    assert torch.count_nonzero(actual[1]) == 0
    assert torch.isfinite(actual).all()


def test_cuda_fused_rope_identical_phase_and_quarter_turn_signs() -> None:
    executor = resolve_compressed_time_local_attention(
        "cuda", compression_ratio=2, window_radius=1
    )
    query = torch.tensor(
        [[[[1.0, 2.0], [-3.0, 4.0], [2.5, -1.0], [-2.0, -3.0]]]],
        device="cuda",
        requires_grad=True,
    )
    key = torch.tensor(
        [[[[5.0, -7.0], [-4.0, 3.0]]]], device="cuda", requires_grad=True
    )
    value = torch.tensor(
        [[[[2.0, 3.0], [-5.0, 7.0]]]], device="cuda", requires_grad=True
    )
    query_valid = torch.ones(1, 4, dtype=torch.bool, device="cuda")
    key_valid = torch.ones(1, 2, dtype=torch.bool, device="cuda")
    common_phase = torch.polar(
        torch.ones(4, 1, 1, device="cuda"),
        torch.full((4, 1, 1), 0.731, device="cuda"),
    )
    common_key_phase = common_phase[:2]

    baseline = executor(
        query,
        key,
        value,
        query_valid=query_valid,
        key_valid=key_valid,
        query_freqs_cis=None,
        key_freqs_cis=None,
    )
    identical = executor(
        query,
        key,
        value,
        query_valid=query_valid,
        key_valid=key_valid,
        query_freqs_cis=common_phase,
        key_freqs_cis=common_key_phase,
    )
    torch.testing.assert_close(identical, baseline, atol=1e-6, rtol=1e-6)

    quarter_turns = torch.polar(
        torch.ones(1, 4, 1, 1, device="cuda"),
        torch.tensor(
            [math.pi / 2, -math.pi / 2, math.pi / 2, -math.pi / 2],
            device="cuda",
        ).reshape(1, 4, 1, 1),
    )
    signed_query = query.detach().expand(1, 2, 4, 2).clone().requires_grad_(True)
    signed_key = key.detach().expand(1, 2, 2, 2).clone().requires_grad_(True)
    signed_value = value.detach().expand(1, 2, 2, 2).clone().requires_grad_(True)
    expected = reference_compressed_time_local_attention(
        _apply_full_rope_explicit(signed_query, quarter_turns),
        signed_key,
        signed_value,
        query_valid=query_valid,
        key_valid=key_valid,
        compression_ratio=2,
        window_radius=1,
    )
    actual = executor(
        signed_query,
        signed_key,
        signed_value,
        query_valid=query_valid,
        key_valid=key_valid,
        query_freqs_cis=quarter_turns,
        key_freqs_cis=torch.ones(2, 1, 1, dtype=torch.complex64, device="cuda"),
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_cuda_rejects_partial_kv_head_count() -> None:
    query = torch.randn(1, 3, 9, 16, device="cuda")
    key = torch.randn(1, 2, 3, 16, device="cuda")
    executor = resolve_compressed_time_local_attention(
        "cuda", compression_ratio=4, window_radius=1
    )

    with pytest.raises(ValueError, match="key heads must be 1 or equal query heads"):
        executor(
            query,
            key,
            key,
            query_valid=torch.ones(1, 9, dtype=torch.bool, device="cuda"),
            key_valid=torch.ones(1, 3, dtype=torch.bool, device="cuda"),
        )


def test_cuda_rejects_compression_ratio_that_does_not_fit_int32() -> None:
    query = torch.randn(1, 1, 1, 16, device="cuda")
    key = torch.randn(1, 1, 1, 16, device="cuda")
    executor = resolve_compressed_time_local_attention(
        "cuda", compression_ratio=2**31, window_radius=0
    )

    with pytest.raises(RuntimeError, match="compression ratio must fit int32"):
        executor(
            query,
            key,
            key,
            query_valid=torch.ones(1, 1, dtype=torch.bool, device="cuda"),
            key_valid=torch.ones(1, 1, dtype=torch.bool, device="cuda"),
        )


def test_cuda_handles_all_invalid() -> None:
    query = torch.randn(1, 2, 9, 16, device="cuda")
    key = torch.randn(1, 2, 3, 16, device="cuda")
    executor = resolve_compressed_time_local_attention(
        "cuda", compression_ratio=4, window_radius=1
    )
    output = executor(
        query,
        key,
        key,
        query_valid=torch.zeros(1, 9, dtype=torch.bool, device="cuda"),
        key_valid=torch.zeros(1, 3, dtype=torch.bool, device="cuda"),
    )
    assert torch.count_nonzero(output) == 0


@pytest.mark.parametrize(
    ("query", "message"),
    [
        pytest.param(
            lambda: torch.randn(1, 2, 9, 32, device="cuda")[..., ::2],
            "unit feature stride",
            id="feature-holey",
        ),
        pytest.param(
            lambda: torch.randn(1, 1, 9, 16, device="cuda").expand(1, 2, 9, 16),
            "positive strides",
            id="overlapping",
        ),
        pytest.param(
            lambda: torch.randn(512, device="cuda").as_strided(
                (1, 2, 9, 16), (512, 16, 8, 1)
            ),
            "supported non-overlapping",
            id="positive-overlapping",
        ),
        pytest.param(
            lambda: torch.randn(1, 2, 16, 9, device="cuda").transpose(2, 3),
            "unit feature stride",
            id="feature-strided",
        ),
    ],
)
def test_cuda_rejects_unsupported_query_layouts(
    query: Callable[[], Tensor],
    message: str,
) -> None:
    query_tensor = query()
    key = torch.randn(1, 2, 3, 16, device="cuda")
    executor = resolve_compressed_time_local_attention(
        "cuda", compression_ratio=4, window_radius=1
    )

    with pytest.raises(RuntimeError, match=message):
        executor(
            query_tensor,
            key,
            key,
            query_valid=torch.ones(1, 9, dtype=torch.bool, device="cuda"),
            key_valid=torch.ones(1, 3, dtype=torch.bool, device="cuda"),
        )


def test_cuda_backward_rejects_unsupported_grad_output_layout() -> None:
    query = torch.randn(1, 9, 2, 16, device="cuda").transpose(1, 2)
    query.requires_grad_(True)
    key = torch.randn(1, 2, 3, 16, device="cuda", requires_grad=True)
    executor = resolve_compressed_time_local_attention(
        "cuda", compression_ratio=4, window_radius=1
    )
    output = executor(
        query,
        key,
        key,
        query_valid=torch.ones(1, 9, dtype=torch.bool, device="cuda"),
        key_valid=torch.ones(1, 3, dtype=torch.bool, device="cuda"),
    )
    grad_output = torch.randn(1, 2, 16, 9, device="cuda").transpose(2, 3)

    with pytest.raises(RuntimeError, match="unit feature stride"):
        output.backward(grad_output)


def test_cuda_rejects_valid_query_without_key_asynchronously() -> None:
    probe = textwrap.dedent(
        """
        import os

        import torch

        from src.utils.models.components.ops.compressed_time_local import (
            resolve_compressed_time_local_attention,
        )

        query = torch.randn(1, 2, 9, 16, device="cuda")
        key = torch.randn(1, 2, 3, 16, device="cuda")
        executor = resolve_compressed_time_local_attention(
            "cuda", compression_ratio=4, window_radius=1
        )
        try:
            executor(
                query,
                key,
                key,
                query_valid=torch.ones(1, 9, dtype=torch.bool, device="cuda"),
                key_valid=torch.zeros(1, 3, dtype=torch.bool, device="cuda"),
            )
            torch.cuda.synchronize()
        except RuntimeError as error:
            if "device-side assert" not in str(error):
                raise
            os._exit(0)
        raise AssertionError("asynchronous invalid-row assertion was not raised")
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr


def test_cuda_rejects_dropout_instead_of_falling_back() -> None:
    query = torch.randn(1, 2, 9, 16, device="cuda")
    key = torch.randn(1, 2, 3, 16, device="cuda")
    executor = resolve_compressed_time_local_attention(
        "cuda", compression_ratio=4, window_radius=1
    )

    with pytest.raises(RuntimeError, match="does not support attention dropout"):
        executor(
            query,
            key,
            key,
            query_valid=torch.ones(1, 9, dtype=torch.bool, device="cuda"),
            key_valid=torch.ones(1, 3, dtype=torch.bool, device="cuda"),
            dropout_p=0.1,
            training=True,
        )
