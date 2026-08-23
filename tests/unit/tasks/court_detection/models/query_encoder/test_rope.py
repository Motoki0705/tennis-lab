"""Tests for positionless-query, patch-only 2-D RoPE."""

from __future__ import annotations

import pytest
import torch

from src.tasks.court_detection.models.query_encoder.rope import apply_patch_only_rope
from src.utils.models.components.rope import RotaryFrequencyComputer


def test_patch_rope_leaves_pose_query_exactly_unmodified() -> None:
    query = torch.arange(80, dtype=torch.float32).reshape(1, 2, 5, 8) / 10.0
    key = query + 1.0
    computer = RotaryFrequencyComputer(dim=8, base=10000.0, n_axes=2)

    rotated_query, rotated_key = apply_patch_only_rope(
        query,
        key,
        grid_hw=(2, 2),
        rope_dim=8,
        frequency_computer=computer,
    )

    torch.testing.assert_close(rotated_query[:, :, 0], query[:, :, 0])
    torch.testing.assert_close(rotated_key[:, :, 0], key[:, :, 0])
    assert not torch.equal(rotated_query[:, :, 2], query[:, :, 2])
    assert not torch.equal(rotated_key[:, :, 2], key[:, :, 2])


def test_patch_rope_rejects_grid_sequence_mismatch() -> None:
    query = torch.zeros(1, 2, 6, 8)
    computer = RotaryFrequencyComputer(dim=8, base=10000.0, n_axes=2)

    with pytest.raises(ValueError, match="one pose query"):
        apply_patch_only_rope(
            query,
            query,
            grid_hw=(2, 2),
            rope_dim=8,
            frequency_computer=computer,
        )
