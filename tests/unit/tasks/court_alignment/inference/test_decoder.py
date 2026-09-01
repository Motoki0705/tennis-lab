"""Unit tests for multi-peak and center-vote Court decoding."""

from __future__ import annotations

from typing import cast

import pytest
import torch

from src.tasks.court_alignment.geometry.court import canonical_court_keypoints
from src.tasks.court_alignment.inference.decoder import (
    CourtInstanceBatch,
    decode_keypoint_peaks,
    group_peak_votes,
)
from src.tasks.court_alignment.inference.predictor import CourtAlignmentPredictor
from src.tasks.court_alignment.models.cnn import CourtAlignmentCNN


def test_decoder_retains_multiple_peaks_per_semantic_channel() -> None:
    logits = torch.full((1, 14, 24, 24), -10.0)
    logits[:, :, 4, 5] = 10.0
    logits[:, :, 18, 19] = 9.0
    votes = torch.zeros(1, 2, 24, 24)

    detections = decode_keypoint_peaks(
        logits,
        votes,
        threshold=0.5,
        nms_kernel=3,
        max_peaks=2,
        subpixel_refine=False,
    )

    assert detections.keypoints_px.shape == (1, 14, 2, 2)
    assert detections.valid.all()
    torch.testing.assert_close(
        detections.keypoints_px[0, 0],
        torch.tensor([[5.0, 4.0], [19.0, 18.0]]),
    )


def test_center_votes_group_peaks_even_when_per_channel_ranks_are_swapped() -> None:
    keypoints = torch.zeros(14, 2, 2)
    votes = torch.zeros_like(keypoints)
    valid = torch.ones(14, 2, dtype=torch.bool)
    scores = torch.zeros(14, 2)
    for channel in range(14):
        keypoints[channel, 0] = torch.tensor([10.0, 10.0]) + channel * 0.1
        keypoints[channel, 1] = torch.tensor([100.0, 100.0]) + channel * 0.1
        votes[channel, 0] = torch.tensor([20.0, 20.0]) - keypoints[channel, 0]
        votes[channel, 1] = torch.tensor([200.0, 200.0]) - keypoints[channel, 1]
        scores[channel] = (
            torch.tensor([0.4, 0.9]) if channel % 2 else torch.tensor([0.9, 0.4])
        )

    grouped = group_peak_votes(
        keypoints,
        votes,
        valid,
        scores,
        cluster_distance_px=3.0,
    )

    assert isinstance(grouped, CourtInstanceBatch)
    assert grouped.num_instances == 2
    torch.testing.assert_close(grouped.centers_px, torch.tensor([[20.0, 20.0], [200.0, 200.0]]))
    assert grouped.valid.all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": -0.1},
        {"nms_kernel": 2},
        {"max_peaks": 0},
    ],
)
def test_decoder_rejects_invalid_options(kwargs: dict[str, object]) -> None:
    logits = torch.zeros(1, 14, 8, 8)
    votes = torch.zeros(1, 2, 8, 8)
    with pytest.raises(ValueError):
        decode_keypoint_peaks(
            logits,
            votes,
            threshold=cast(float, kwargs.get("threshold", 0.5)),
            nms_kernel=cast(int, kwargs.get("nms_kernel", 3)),
            max_peaks=cast(int, kwargs.get("max_peaks", 2)),
        )


def test_predictor_rejects_heatmap_values_outside_contract() -> None:
    predictor = CourtAlignmentPredictor(CourtAlignmentCNN(base_channels=4))
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        predictor.predict(torch.full((1, 1, 16, 16), 1.1))


def test_cluster_quality_is_ranked_before_max_instance_truncation() -> None:
    keypoints = torch.zeros(14, 2, 2)
    votes = torch.zeros_like(keypoints)
    valid = torch.zeros(14, 2, dtype=torch.bool)
    scores = torch.zeros(14, 2)
    court = canonical_court_keypoints() * 2.0 + torch.tensor([180.0, 160.0])
    for channel in range(14):
        keypoints[channel, 1] = court[channel]
        votes[channel, 1] = torch.tensor([180.0, 160.0]) - court[channel]
        valid[channel, 1] = True
        scores[channel, 1] = 0.5
    for channel in range(2):
        keypoints[channel, 0] = torch.tensor([10.0 + channel, 10.0])
        votes[channel, 0] = torch.tensor([20.0, 20.0]) - keypoints[channel, 0]
        valid[channel, 0] = True
        scores[channel, 0] = 0.99

    grouped = group_peak_votes(
        keypoints,
        votes,
        valid,
        scores,
        cluster_distance_px=2.0,
        max_instances=1,
    )

    assert isinstance(grouped, CourtInstanceBatch)
    assert grouped.semantic_count is not None
    assert grouped.aggregate_confidence is not None
    assert grouped.geometry_residual_px is not None
    assert grouped.semantic_count.tolist() == [14]
    torch.testing.assert_close(grouped.centers_px[0], torch.tensor([180.0, 160.0]))
    assert float(grouped.aggregate_confidence[0]) == pytest.approx(7.0)
    assert float(grouped.geometry_residual_px[0]) < 1.0e-4


def test_geometry_residual_breaks_equal_count_and_confidence_ties() -> None:
    canonical = canonical_court_keypoints()
    good = canonical * 2.0 + torch.tensor([150.0, 120.0])
    distorted = canonical * 2.0 + torch.tensor([40.0, 40.0])
    distorted[0] += torch.tensor([12.0, -7.0])
    keypoints = torch.stack((distorted, good), dim=1)
    centers = torch.tensor(([40.0, 40.0], [150.0, 120.0]))
    votes = centers[None] - keypoints
    valid = torch.ones(14, 2, dtype=torch.bool)
    scores = torch.full((14, 2), 0.6)

    grouped = group_peak_votes(
        keypoints,
        votes,
        valid,
        scores,
        cluster_distance_px=2.0,
        max_instances=1,
    )

    assert isinstance(grouped, CourtInstanceBatch)
    assert grouped.geometry_residual_px is not None
    torch.testing.assert_close(grouped.centers_px[0], centers[1])
    assert float(grouped.geometry_residual_px[0]) < 1.0e-4
