"""Unit tests for :mod:`src.utils.geometry.bbox`."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.geometry.bbox import bbox_max_side_ratio, pairwise_iou


class TestBboxMaxSideRatio:
    def test_picks_the_more_dominant_side(self) -> None:
        # width spans 50% of the frame, height spans 10%; the max wins.
        assert bbox_max_side_ratio(100.0, 20.0, 200.0, 200.0) == pytest.approx(0.5)

    def test_height_can_be_the_dominant_side(self) -> None:
        assert bbox_max_side_ratio(10.0, 80.0, 100.0, 100.0) == pytest.approx(0.8)

    def test_normalized_boxes_use_unit_image_size(self) -> None:
        # YOLO-style normalized w/h: ratio is just max(w, h).
        assert bbox_max_side_ratio(0.3, 0.12, 1.0, 1.0) == pytest.approx(0.3)

    def test_full_frame_box_is_one(self) -> None:
        assert bbox_max_side_ratio(640.0, 480.0, 640.0, 480.0) == pytest.approx(1.0)

    def test_non_positive_image_size_raises(self) -> None:
        with pytest.raises(ValueError):
            bbox_max_side_ratio(1.0, 1.0, 0.0, 10.0)


def test_pairwise_iou_known_values_and_empty_boxes() -> None:
    first = np.array([[0.0, 0, 10, 10], [5, 5, 5, 5]])
    second = np.array([[0.0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    iou = pairwise_iou(first, second)
    np.testing.assert_allclose(iou[0], [1.0, 50 / 150, 0.0])
    np.testing.assert_allclose(iou[1], 0.0)  # a degenerate box overlaps nothing
    with pytest.raises(ValueError):
        pairwise_iou(first[:, :3], second)
