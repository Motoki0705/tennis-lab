"""Queued-GPU tests for fused compressed time-local CUDA attention."""

from __future__ import annotations

import os

import pytest
import torch
from torch import Tensor

from src.utils.models.components.ops.compressed_time_local import (
    reference_compressed_time_local_attention,
    resolve_compressed_time_local_attention,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("TENNIS_LAB_RUN_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
    reason="CUDA operation tests run only in an attributed training-queue job",
)


def _inputs(dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(753)
    query = torch.randn(2, 3, 37, 128, device="cuda", dtype=dtype, generator=generator)[
        ..., ::2
    ]
    key = torch.randn(2, 3, 10, 128, device="cuda", dtype=dtype, generator=generator)[
        ..., ::2
    ]
    value = torch.randn(2, 3, 10, 128, device="cuda", dtype=dtype, generator=generator)[
        ..., ::2
    ]
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


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.float32, 2.0e-5, 2.0e-4),
        (torch.bfloat16, 6.5e-2, 2.0e-2),
    ],
)
def test_cuda_matches_reference_forward_and_all_qkv_gradients(
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> None:
    inputs = _inputs(dtype)
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
    upstream = torch.randn_like(reference)
    reference.backward(upstream)
    actual.backward(upstream)

    torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
    for actual_input, reference_input in zip(
        cuda_inputs, reference_inputs, strict=True
    ):
        torch.testing.assert_close(
            actual_input.grad,
            reference_input.grad,
            atol=atol,
            rtol=rtol,
        )
    assert torch.count_nonzero(actual[1]) == 0


def test_cuda_handles_all_invalid_and_rejects_valid_query_without_key() -> None:
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

    with pytest.raises(RuntimeError, match="no valid compressed key"):
        executor(
            query,
            key,
            key,
            query_valid=torch.ones(1, 9, dtype=torch.bool, device="cuda"),
            key_valid=torch.zeros(1, 3, dtype=torch.bool, device="cuda"),
        )


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
