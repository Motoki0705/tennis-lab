"""Unit tests for the full-resolution Court Alignment CNN."""

from __future__ import annotations

import pytest
import torch

from src.tasks.court_alignment.models.cnn import (
    CourtAlignmentCNN,
    CourtAlignmentModelOutput,
    validate_court_alignment_input,
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
        validate_court_alignment_input(input_tensor)


def test_invalid_keypoint_count_and_group_count_fail_fast() -> None:
    with pytest.raises(ValueError, match="fourteen"):
        CourtAlignmentCNN(num_keypoints=13)
    with pytest.raises(ValueError, match="group_norm_groups"):
        CourtAlignmentCNN(group_norm_groups=0)


def test_compiled_forward_has_no_scalar_graph_break() -> None:
    model = torch.compile(
        CourtAlignmentCNN(base_channels=4, group_norm_groups=2),
        backend="eager",
        fullgraph=True,
    )
    output = model(torch.zeros(1, 1, 16, 16))
    assert output.heatmap_logits.shape == (1, 14, 16, 16)


def test_encoder_has_four_downsamples_and_initial_heatmap_prior() -> None:
    model = CourtAlignmentCNN(
        base_channels=4,
        group_norm_groups=2,
        heatmap_prior_probability=0.1,
    )

    assert all(hasattr(model, name) for name in ("down1", "down2", "down3", "down4"))
    output_projection = model.head[-1]
    assert isinstance(output_projection, torch.nn.Conv2d)
    assert output_projection.bias is not None
    torch.testing.assert_close(
        output_projection.bias[:14].sigmoid(),
        torch.full((14,), 0.1),
        atol=1.0e-6,
        rtol=0.0,
    )
    assert model.num_downsamples == 4
    assert model.receptive_field_px >= 181  # corner-to-centre for 256x256
    torch.testing.assert_close(
        output_projection.bias[14:],
        torch.zeros(2),
        atol=0.0,
        rtol=0.0,
    )


def test_heatmap_prior_probability_is_strict() -> None:
    with pytest.raises(ValueError, match="heatmap_prior_probability"):
        CourtAlignmentCNN(heatmap_prior_probability=1.0)
