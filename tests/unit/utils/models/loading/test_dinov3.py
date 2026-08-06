"""Tests for DINOv3 dynamic-output boundary validation."""

from __future__ import annotations

import pytest
import torch

from src.utils.models.loading import require_dinov3_patch_tokens

pytestmark = pytest.mark.unit


def test_require_dinov3_patch_tokens_accepts_exact_contract() -> None:
    tokens = torch.randn(2, 16, 8)

    result = require_dinov3_patch_tokens(
        {"x_norm_patchtokens": tokens},
        expected_batch_size=2,
        expected_num_tokens=16,
        expected_embed_dim=8,
    )

    assert result is tokens


@pytest.mark.parametrize(
    ("outputs", "error", "message"),
    [
        ([], TypeError, "must return a mapping"),
        ({}, KeyError, "missing required"),
        ({"x_norm_patchtokens": []}, TypeError, "must be a tensor"),
        ({"x_norm_patchtokens": torch.randn(2, 8)}, ValueError, "shape"),
        ({"x_norm_patchtokens": torch.randn(1, 16, 8)}, ValueError, "batch size"),
        ({"x_norm_patchtokens": torch.randn(2, 15, 8)}, ValueError, "token count"),
        ({"x_norm_patchtokens": torch.randn(2, 16, 7)}, ValueError, "embedding width"),
    ],
)
def test_require_dinov3_patch_tokens_rejects_contract_violations(
    outputs: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        require_dinov3_patch_tokens(
            outputs,
            expected_batch_size=2,
            expected_num_tokens=16,
            expected_embed_dim=8,
        )
