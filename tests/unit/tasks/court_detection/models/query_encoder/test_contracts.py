"""Tests for strict query-encoder tensor contracts."""

from __future__ import annotations

import pytest
import torch

from src.tasks.court_detection.models.query_encoder.contracts import (
    COURT_POSE10D_RAW_ORDER,
    CourtPose10DRaw,
    PatchTokenBatch,
)


def test_patch_boundary_records_exact_grid_and_minimal_padding() -> None:
    batch = PatchTokenBatch(
        tokens=torch.zeros(2, 25, 8),
        original_hw=(17, 19),
        padded_hw=(20, 20),
        padding_hw=(3, 1),
        grid_hw=(5, 5),
        patch_size=4,
    )

    assert batch.batch_size == 2
    assert batch.embed_dim == 8


@pytest.mark.parametrize(
    ("token_count", "padding_hw"),
    [(26, (3, 1)), (25, (7, 1))],
)
def test_patch_boundary_rejects_special_tokens_or_nonminimal_padding(
    token_count: int,
    padding_hw: tuple[int, int],
) -> None:
    with pytest.raises(ValueError):
        PatchTokenBatch(
            tokens=torch.zeros(1, token_count, 8),
            original_hw=(17, 19),
            padded_hw=(20, 20),
            padding_hw=padding_hw,
            grid_hw=(5, 5),
            patch_size=4,
        )


def test_pose10d_contract_has_exact_shape_and_scalar_order() -> None:
    raw = CourtPose10DRaw(torch.arange(20, dtype=torch.float32).reshape(2, 10))

    assert raw.values.shape == (2, 10)
    assert COURT_POSE10D_RAW_ORDER == (
        "tx",
        "ty",
        "tz",
        "a11",
        "a12",
        "a13",
        "a21",
        "a22",
        "a23",
        "logf",
    )

    with pytest.raises(ValueError, match="exact shape"):
        CourtPose10DRaw(torch.zeros(2, 11))
