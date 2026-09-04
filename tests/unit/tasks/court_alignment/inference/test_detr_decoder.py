"""Tests for DINO multi-court query decoding."""

from __future__ import annotations

import math

import pytest
import torch

from src.tasks.court_alignment.geometry.court import COURT_LENGTH_M
from src.tasks.court_alignment.inference.detr_decoder import decode_detr_courts


def _raw_for_rotation(rotation_deg: float, long_normalized: float) -> torch.Tensor:
    rotation = math.radians(rotation_deg)
    long_angle = rotation + math.pi / 2.0
    return torch.tensor(
        [
            torch.logit(torch.tensor(long_normalized)),
            math.cos(2.0 * long_angle),
            math.sin(2.0 * long_angle),
        ]
    )


@pytest.mark.parametrize("rotation_deg", [0.0, 90.0, 179.0])
def test_decode_recovers_rotation_modulo_pi(rotation_deg: float) -> None:
    logits = torch.tensor([[[8.0]]])
    boxes = torch.tensor([[[0.25, 0.75, 0.2, 0.4]]])
    court = _raw_for_rotation(rotation_deg, 0.5)[None, None]

    decoded = decode_detr_courts(logits, boxes, court, image_size=(800, 800))
    sample = decoded[0]

    assert sample.num_instances == 1
    torch.testing.assert_close(sample.translation_px[0], torch.tensor([200.0, 600.0]))
    assert float(sample.long_sides_px[0]) == pytest.approx(400.0)
    assert float(sample.scale_px_per_metre[0]) == pytest.approx(400.0 / COURT_LENGTH_M)
    assert math.degrees(float(sample.rotation_rad[0])) == pytest.approx(
        rotation_deg, abs=1.0e-4
    )
    assert sample.corners_px.shape == (1, 4, 2)


def test_decoder_keeps_multiple_overlapping_queries_without_nms() -> None:
    logits = torch.tensor([[[4.0], [3.0], [-4.0]]])
    boxes = torch.tensor(
        [[[0.5, 0.5, 0.4, 0.5], [0.5, 0.5, 0.4, 0.5], [0.8, 0.8, 0.1, 0.1]]]
    )
    court = torch.stack(
        (
            _raw_for_rotation(10.0, 0.4),
            _raw_for_rotation(12.0, 0.4),
            _raw_for_rotation(80.0, 0.2),
        )
    )[None]

    decoded = decode_detr_courts(
        logits,
        boxes,
        court,
        image_size=(800, 800),
        threshold=0.5,
    )

    assert decoded.num_instances.tolist() == [2]
    assert decoded[0].query_indices.tolist() == [0, 1]
    torch.testing.assert_close(decoded[0].centers_px[0], decoded[0].centers_px[1])


def test_decoder_threshold_and_top_k_rank_queries_by_court_score() -> None:
    logits = torch.tensor([[[1.0], [3.0], [2.0]]])
    boxes = torch.full((1, 3, 4), 0.5)
    court = torch.stack([_raw_for_rotation(0.0, 0.3)] * 3)[None]

    decoded = decode_detr_courts(
        logits,
        boxes,
        court,
        image_size=(800, 800),
        threshold=0.7,
        top_k=2,
    )

    assert decoded[0].query_indices.tolist() == [1, 2]
    assert decoded[0].scores.tolist() == sorted(
        decoded[0].scores.tolist(), reverse=True
    )


def test_decoder_returns_typed_empty_sample() -> None:
    decoded = decode_detr_courts(
        torch.full((1, 2, 1), -10.0),
        torch.full((1, 2, 4), 0.5),
        torch.zeros((1, 2, 3)),
        image_size=(800, 800),
        threshold=0.5,
    )

    assert decoded[0].num_instances == 0
    assert decoded[0].corners_px.shape == (0, 4, 2)
    assert decoded[0].rotation_rad.shape == (0,)


def test_rectangular_decode_uses_width_height_and_max_side_independently() -> None:
    decoded = decode_detr_courts(
        torch.tensor([[[8.0]]]),
        torch.tensor([[[0.25, 0.75, 0.2, 0.4]]]),
        _raw_for_rotation(30.0, 0.5)[None, None],
        image_size=(400, 800),
    )[0]

    torch.testing.assert_close(decoded.centers_px[0], torch.tensor([200.0, 300.0]))
    assert float(decoded.long_sides_px[0]) == pytest.approx(400.0)
    assert float(decoded.short_sides_px[0]) == pytest.approx(400.0 * 10.97 / 23.77)
    assert math.degrees(float(decoded.rotation_rad[0])) == pytest.approx(
        30.0, abs=1.0e-4
    )
