from __future__ import annotations

import pytest
import torch

from src.utils.models.components.ops.loader import is_time_local_cuda_available
from src.utils.models.components.ops.time_local import (
    reference_time_local_attention,
    time_local_attention,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available() or not is_time_local_cuda_available(),
        reason="CUDA device or time-local CUDA extension is not available",
    ),
]


def test_time_local_attention_cuda_matches_reference_outputs() -> None:
    query = torch.randn(2, 4, 8, 16, device="cuda", dtype=torch.float32)
    key = torch.randn(2, 4, 8, 16, device="cuda", dtype=torch.float32)
    value = torch.randn(2, 4, 8, 16, device="cuda", dtype=torch.float32)
    valid_mask = torch.tensor(
        [
            [True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, False, False],
        ],
        device="cuda",
        dtype=torch.bool,
    )

    actual = time_local_attention(
        query,
        key,
        value,
        valid_mask=valid_mask,
        window_radius=2,
        use_cuda=True,
    )
    expected = reference_time_local_attention(
        query,
        key,
        value,
        valid_mask=valid_mask,
        window_radius=2,
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_time_local_attention_cuda_matches_reference_gradients() -> None:
    valid_mask = torch.tensor(
        [[True, True, True, True, True, True, False, False]],
        device="cuda",
        dtype=torch.bool,
    )

    base_query = torch.randn(1, 2, 8, 8, device="cuda", dtype=torch.float32)
    base_key = torch.randn(1, 2, 8, 8, device="cuda", dtype=torch.float32)
    base_value = torch.randn(1, 2, 8, 8, device="cuda", dtype=torch.float32)

    query_cuda = base_query.clone().requires_grad_(True)
    key_cuda = base_key.clone().requires_grad_(True)
    value_cuda = base_value.clone().requires_grad_(True)
    loss_cuda = time_local_attention(
        query_cuda,
        key_cuda,
        value_cuda,
        valid_mask=valid_mask,
        window_radius=2,
        use_cuda=True,
    ).sum()
    loss_cuda.backward()

    query_ref = base_query.clone().requires_grad_(True)
    key_ref = base_key.clone().requires_grad_(True)
    value_ref = base_value.clone().requires_grad_(True)
    loss_ref = reference_time_local_attention(
        query_ref,
        key_ref,
        value_ref,
        valid_mask=valid_mask,
        window_radius=2,
    ).sum()
    loss_ref.backward()

    torch.testing.assert_close(query_cuda.grad, query_ref.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(key_cuda.grad, key_ref.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(value_cuda.grad, value_ref.grad, atol=1e-5, rtol=1e-5)