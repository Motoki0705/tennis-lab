from __future__ import annotations

import torch
import torch.nn.functional as F

from src.utils.models.components.ops.time_local import (
    build_local_attention_keep_mask,
    reference_time_local_attention,
)


def test_sliding_window_reference_matches_dense_mask_sdpa() -> None:
    query = torch.randn(2, 3, 5, 4)
    key = torch.randn(2, 3, 5, 4)
    value = torch.randn(2, 3, 5, 4)
    valid_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [False, True, True, True, True],
        ],
        dtype=torch.bool,
    )

    actual = reference_time_local_attention(
        query,
        key,
        value,
        valid_mask=valid_mask,
        window_radius=1,
    )

    expected_mask = build_local_attention_keep_mask(valid_mask, window_radius=1)
    expected = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=expected_mask[:, None, :, :],
        is_causal=False,
    )
    torch.testing.assert_close(actual, expected)


def test_sliding_window_falls_back_to_global_valid_keys() -> None:
    valid_mask = torch.tensor([[False, False, True, False]], dtype=torch.bool)

    actual = build_local_attention_keep_mask(valid_mask, window_radius=0)
    expected = torch.tensor(
        [
            [
                [False, False, True, False],
                [False, False, True, False],
                [False, False, True, False],
                [False, False, True, False],
            ]
        ],
        dtype=torch.bool,
    )
    assert torch.equal(actual, expected)