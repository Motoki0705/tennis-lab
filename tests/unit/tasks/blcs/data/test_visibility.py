"""Tests for the persisted-observation visibility boundary."""

from __future__ import annotations

import pytest
import torch

from src.tasks.blcs.data.visibility import zero_invisible_uv


def test_zero_invisible_uv_removes_arbitrary_hidden_values_only() -> None:
    uv = torch.tensor(
        [
            [0.25, 0.75],
            [float("nan"), float("inf")],
            [-8.0, 12.0],
        ]
    )
    visibility = torch.tensor([True, False, False])

    result = zero_invisible_uv(uv, visibility)

    torch.testing.assert_close(result[0], uv[0])
    torch.testing.assert_close(result[1:], torch.zeros(2, 2))
    assert torch.isnan(uv[1, 0])
    assert uv[2, 0] == -8.0


def test_zero_invisible_uv_preserves_invalid_visible_values_for_strict_validation() -> None:
    uv = torch.tensor([[1.25, -0.25]])

    result = zero_invisible_uv(uv, torch.tensor([True]))

    torch.testing.assert_close(result, uv)


@pytest.mark.parametrize(
    ("uv", "visibility", "message"),
    [
        (torch.zeros(2, 3), torch.zeros(2), "coordinate axis of size 2"),
        (torch.zeros(2, 2), torch.zeros(3), "visibility shape must match"),
    ],
)
def test_zero_invisible_uv_rejects_shape_mismatch(
    uv: torch.Tensor,
    visibility: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        zero_invisible_uv(uv, visibility)
