"""Tests for PLCS output-head temporal decomposition."""

from __future__ import annotations

import torch

from src.tasks.plcs.models.components.heads import (
    TemporalDecomposedCanonicalPoseHead,
)


def _head() -> TemporalDecomposedCanonicalPoseHead:
    torch.manual_seed(7)
    head = TemporalDecomposedCanonicalPoseHead(
        input_dim=8,
        hidden_dim=4,
        num_layers=1,
        dropout=0.0,
        num_keypoints=3,
    )
    head.eval()
    return head


def test_temporal_decomposition_preserves_static_mean_and_ignores_padding() -> None:
    head = _head()
    features = torch.randn(2, 5, 8)
    frame_valid = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, True]]
    )

    output = head(features, frame_valid)
    changed_padding = features.clone()
    changed_padding[0, 3:] = 10_000.0
    changed_output = head(changed_padding, frame_valid)

    assert output.shape == (2, 5, 3, 3)
    torch.testing.assert_close(output[0, :3], changed_output[0, :3])

    weight = frame_valid.to(features.dtype).unsqueeze(-1)
    mean_features = (features * weight).sum(1) / weight.sum(1)
    expected_static = head.static_head(mean_features)
    for batch_idx in range(2):
        valid_output = output[batch_idx, frame_valid[batch_idx]]
        torch.testing.assert_close(valid_output.mean(0), expected_static[batch_idx])
