"""Unit tests for the full-resolution Court Alignment CNN."""

from __future__ import annotations

import pytest
import torch

from src.tasks.court_alignment.models.cnn import (
    CourtAlignmentCNN,
    CourtAlignmentModelOutput,
)


def test_forward_preserves_spatial_resolution_and_contract() -> None:
    model = CourtAlignmentCNN(base_channels=8)
    output = model(torch.zeros(2, 1, 31, 35))

    assert isinstance(output, CourtAlignmentModelOutput)
    assert output.heatmap_logits.shape == (2, 14, 31, 35)
    assert output.center_votes.shape == (2, 2, 31, 35)
    torch.testing.assert_close(output["heatmap_logits"], output.heatmap_logits)
    assert list(output) == ["heatmap_logits", "center_votes"]


@pytest.mark.parametrize(
    ("input_tensor", "error"),
    [
        (torch.zeros(1, 1, 8), "shape"),
        (torch.zeros(1, 2, 8, 8), "shape"),
        (torch.zeros(1, 1, 8, 8, dtype=torch.int64), "floating"),
    ],
)
def test_forward_rejects_invalid_input_shape_or_dtype(
    input_tensor: torch.Tensor, error: str
) -> None:
    with pytest.raises((ValueError, TypeError), match=error):
        CourtAlignmentCNN(base_channels=8)(input_tensor)


def test_invalid_keypoint_count_and_group_count_fail_fast() -> None:
    with pytest.raises(ValueError, match="fourteen"):
        CourtAlignmentCNN(num_keypoints=13)
    with pytest.raises(ValueError, match="group_norm_groups"):
        CourtAlignmentCNN(group_norm_groups=0)
