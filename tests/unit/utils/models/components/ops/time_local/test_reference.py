"""Tests for prepared-tensor reference time-local attention."""

from __future__ import annotations

import pytest
import torch

from src.utils.models.components.ops.time_local import api as time_local_api
from src.utils.models.components.ops.time_local.api import resolve_time_local_attention
from src.utils.models.components.ops.time_local.layout import (
    build_local_attention_keep_mask,
    normalize_valid_mask,
)
from src.utils.models.components.ops.time_local.reference import (
    reference_time_local_attention,
)


def test_local_mask_uses_global_valid_fallback_for_empty_rows() -> None:
    valid_mask = normalize_valid_mask(
        torch.tensor([[True, False, False, False]], dtype=torch.bool)
    )

    keep_mask = build_local_attention_keep_mask(valid_mask, window_radius=1)

    assert keep_mask.shape == (1, 4, 4)
    assert keep_mask.any(dim=-1).all()
    assert keep_mask[0, 3, 0]


def test_reference_attention_consumes_prepared_tensors() -> None:
    query = torch.randn(2, 3, 5, 4)
    valid_mask = normalize_valid_mask(
        torch.tensor(
            [
                [True, True, True, False, False],
                [False, False, False, False, False],
            ]
        )
    )

    output = reference_time_local_attention(
        query,
        query,
        query,
        attn_mask=build_local_attention_keep_mask(valid_mask, window_radius=1),
    )

    assert output.shape == query.shape
    assert torch.isfinite(output).all()


def test_backend_is_resolved_before_tensor_execution() -> None:
    executor = resolve_time_local_attention("reference", window_radius=1)

    assert executor is reference_time_local_attention


def test_cuda_backend_is_resolved_before_tensor_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable() -> None:
        raise RuntimeError("extension unavailable at composition")

    monkeypatch.setattr(
        time_local_api,
        "require_time_local_cuda_extension",
        unavailable,
    )

    with pytest.raises(RuntimeError, match="at composition"):
        resolve_time_local_attention("cuda", window_radius=1)


@pytest.mark.parametrize("window_radius", [-1, 1.0, True])
def test_window_policy_is_validated_during_resolution(window_radius: object) -> None:
    with pytest.raises(ValueError, match="non-negative int"):
        resolve_time_local_attention(
            "reference",
            window_radius=window_radius,  # type: ignore[arg-type]
        )
