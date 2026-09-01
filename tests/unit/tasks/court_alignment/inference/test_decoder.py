"""Unit tests for multi-peak and center-vote Court decoding."""

from __future__ import annotations

from typing import cast

import pytest
import torch

from src.tasks.court_alignment.inference.decoder import (
    CourtInstanceBatch,
    decode_keypoint_peaks,
    group_peak_votes,
)


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
