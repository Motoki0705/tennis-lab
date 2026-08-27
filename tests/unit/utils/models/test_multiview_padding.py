"""Tests for fixed-query masks derived from multi-view padding."""

from __future__ import annotations

import pytest
import torch

from src.utils.models import build_fixed_query_padding_masks
from src.utils.models.multiview_padding import (
    build_compressed_spatial_attention_keep_mask,
)


def _expected_dense_keep_mask(valid: torch.Tensor) -> torch.Tensor:
    fixed = valid.clone()
    fixed[~fixed.any(dim=1), 0] = True
    return fixed[:, None, :].expand(-1, fixed.shape[1], -1)


def test_nonrectangular_padding_builds_raw_validity_and_dense_keep_masks() -> None:
    padding_mask = torch.tensor(
        [
            [
                [False, True, False],
                [True, True, True],
            ]
        ]
    )

    masks = build_fixed_query_padding_masks(padding_mask, num_queries=2)

    context_valid = ~padding_mask
    frame_valid = torch.tensor([[True, False, True]])
    object_state_valid = context_valid.unsqueeze(-1).expand(1, 2, 3, 2)
    spatial_valid = torch.tensor(
        [
            [True, True, True, True, False, False],
            [False, False, False, False, False, False],
            [True, True, True, True, False, False],
        ]
    )
    object_temporal_valid = torch.tensor(
        [
            [True, False, True],
            [False, False, False],
        ]
    )
    query_temporal_valid = frame_valid.expand(2, -1)

    assert torch.equal(masks.context_valid, context_valid)
    assert torch.equal(masks.frame_valid, frame_valid)
    assert torch.equal(masks.object_state_valid, object_state_valid)
    assert torch.equal(
        masks.spatial_attention_keep_mask,
        _expected_dense_keep_mask(spatial_valid),
    )
    assert torch.equal(
        masks.object_temporal_state_valid,
        object_temporal_valid,
    )
    assert torch.equal(
        masks.object_temporal_attention_keep_mask,
        _expected_dense_keep_mask(object_temporal_valid),
    )
    assert torch.equal(
        masks.query_temporal_state_valid,
        query_temporal_valid,
    )
    assert torch.equal(
        masks.query_temporal_attention_keep_mask,
        _expected_dense_keep_mask(query_temporal_valid),
    )


def test_outputs_retain_padding_mask_device_and_boolean_dtype() -> None:
    padding_mask = torch.zeros(2, 3, 4, dtype=torch.bool)

    masks = build_fixed_query_padding_masks(padding_mask, num_queries=5)

    for value in (
        masks.context_valid,
        masks.frame_valid,
        masks.object_state_valid,
        masks.spatial_attention_keep_mask,
        masks.object_temporal_state_valid,
        masks.object_temporal_attention_keep_mask,
        masks.query_temporal_state_valid,
        masks.query_temporal_attention_keep_mask,
    ):
        assert value.dtype == torch.bool
        assert value.device == padding_mask.device


def test_compressed_spatial_mask_uses_exact_q_plus_v_width() -> None:
    padding_mask = torch.tensor(
        [[[False, True, True], [True, False, True], [True, True, True]]]
    )

    keep_mask = build_compressed_spatial_attention_keep_mask(
        padding_mask,
        num_queries=4,
    )

    assert keep_mask.shape == (3, 7, 7)
    expected_valid = torch.tensor(
        [
            [True, True, True, True, True, False, False],
            [True, True, True, True, False, True, False],
            [False, False, False, False, False, False, False],
        ]
    )
    assert torch.equal(keep_mask, _expected_dense_keep_mask(expected_valid))


def test_compressed_spatial_mask_repairs_only_all_padding_attention_rows() -> None:
    padding_mask = torch.ones(2, 3, 2, dtype=torch.bool)

    keep_mask = build_compressed_spatial_attention_keep_mask(
        padding_mask,
        num_queries=4,
    )

    assert keep_mask.shape == (4, 7, 7)
    assert keep_mask[..., 0].all()
    assert not keep_mask[..., 1:].any()


@pytest.mark.parametrize(
    "padding_mask",
    [
        torch.zeros(2, 3, dtype=torch.bool),
        torch.zeros(1, 2, 3, 4, dtype=torch.bool),
    ],
)
def test_rejects_padding_mask_with_wrong_rank(padding_mask: torch.Tensor) -> None:
    with pytest.raises(ValueError, match=r"shape \(B,V,T\)"):
        build_fixed_query_padding_masks(padding_mask, num_queries=2)


def test_rejects_non_boolean_padding_mask() -> None:
    with pytest.raises(TypeError, match="torch.bool"):
        build_fixed_query_padding_masks(torch.zeros(1, 2, 3), num_queries=2)


@pytest.mark.parametrize(
    "shape",
    [
        (0, 2, 3),
        (1, 0, 3),
        (1, 2, 0),
    ],
)
def test_rejects_empty_padding_axis(shape: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError, match="must all be nonempty"):
        build_fixed_query_padding_masks(
            torch.zeros(shape, dtype=torch.bool),
            num_queries=2,
        )


@pytest.mark.parametrize("num_queries", [0, -1])
def test_rejects_nonpositive_query_width(num_queries: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        build_fixed_query_padding_masks(
            torch.zeros(1, 2, 3, dtype=torch.bool),
            num_queries=num_queries,
        )


@pytest.mark.parametrize("num_queries", [True, 2.0])
def test_rejects_non_integer_query_width(num_queries: object) -> None:
    with pytest.raises(TypeError, match="exactly int"):
        build_fixed_query_padding_masks(
            torch.zeros(1, 2, 3, dtype=torch.bool),
            num_queries=num_queries,  # type: ignore[arg-type]
        )
