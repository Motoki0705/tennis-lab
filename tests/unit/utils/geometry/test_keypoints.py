"""Unit tests for :mod:`src.utils.geometry.keypoints`."""

from __future__ import annotations

import numpy as np

from src.utils.geometry.keypoints import (
    clamp_pixel_coordinate,
    denormalize_keypoints,
    normalize_keypoints,
)


class TestClampPixelCoordinate:
    def test_value_inside_range_is_unchanged(self) -> None:
        assert clamp_pixel_coordinate(5.0, 10) == 5.0

    def test_negative_value_clamped_to_zero(self) -> None:
        assert clamp_pixel_coordinate(-3.0, 10) == 0.0

    def test_value_above_max_clamped_to_axis_size_minus_one(self) -> None:
        assert clamp_pixel_coordinate(100.0, 10) == 9.0

    def test_zero_axis_size_clamps_to_zero(self) -> None:
        # max(axis_size - 1, 0) keeps the upper bound non-negative.
        assert clamp_pixel_coordinate(5.0, 0) == 0.0

    def test_returns_float(self) -> None:
        assert isinstance(clamp_pixel_coordinate(3, 10), float)


class TestNormalizeKeypoints:
    def test_scales_by_width_and_height(self) -> None:
        kp = np.array([[100.0, 50.0], [200.0, 100.0]], dtype=np.float32)
        out = normalize_keypoints(kp, width=200, height=100)
        expected = np.array([[0.5, 0.5], [1.0, 1.0]], dtype=np.float32)
        np.testing.assert_allclose(out, expected)

    def test_does_not_mutate_input(self) -> None:
        kp = np.array([[100.0, 50.0]], dtype=np.float32)
        original = kp.copy()
        normalize_keypoints(kp, width=200, height=100)
        np.testing.assert_array_equal(kp, original)

    def test_supports_leading_dims(self) -> None:
        kp: np.ndarray = np.zeros((4, 7, 2), dtype=np.float32)
        assert normalize_keypoints(kp, 10, 10).shape == (4, 7, 2)


class TestDenormalizeKeypoints:
    def test_scales_back_to_pixels(self) -> None:
        kp = np.array([[0.5, 0.5]], dtype=np.float32)
        out = denormalize_keypoints(kp, width=200, height=100)
        np.testing.assert_allclose(out, np.array([[100.0, 50.0]], dtype=np.float32))

    def test_round_trip_recovers_original(self) -> None:
        kp = np.array([[123.0, 45.0], [12.0, 99.0]], dtype=np.float32)
        normalized = normalize_keypoints(kp, 256, 128)
        recovered = denormalize_keypoints(normalized, 256, 128)
        np.testing.assert_allclose(recovered, kp, rtol=1e-5)

    def test_does_not_mutate_input(self) -> None:
        kp = np.array([[0.5, 0.5]], dtype=np.float32)
        original = kp.copy()
        denormalize_keypoints(kp, 200, 100)
        np.testing.assert_array_equal(kp, original)
