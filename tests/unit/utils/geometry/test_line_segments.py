from __future__ import annotations

import cv2
import numpy as np

from src.utils.geometry.line_segments import (
    RansacLineConfig,
    canonicalize_segment,
    extract_line_segments,
    sort_and_pad_segments,
)


def _config(*, max_lines: int = 6) -> RansacLineConfig:
    return RansacLineConfig(
        max_iterations=300,
        distance_threshold_px=1.5,
        min_inliers=15,
        min_segment_length_px=10.0,
        max_lines=max_lines,
        skeletonize=False,
        min_component_size=5,
        max_points=2000,
    )


def test_extracts_multiple_finite_lines_with_outliers() -> None:
    line_map: np.ndarray[tuple[int, int], np.dtype[np.uint8]] = np.zeros(
        (96, 160), dtype=np.uint8
    )
    cv2.line(line_map, (10, 20), (145, 20), 255, 1)
    cv2.line(line_map, (30, 8), (30, 85), 255, 1)
    cv2.line(line_map, (60, 80), (140, 35), 255, 1)
    rng = np.random.default_rng(2)
    ys = rng.integers(0, 96, size=80)
    xs = rng.integers(0, 160, size=80)
    line_map[ys, xs] = 255

    result = extract_line_segments(
        line_map,
        config=_config(),
        rng=np.random.default_rng(11),
    )

    assert result.segments.shape == (6, 4)
    assert result.diagnostics.extracted_line_count >= 3
    valid = result.segments[np.any(result.segments != 0, axis=1)]
    assert len(valid) >= 3
    assert np.all((valid >= 0.0) & (valid <= 1.0))
    assert not np.any(np.all(valid == 0.0, axis=1))


def test_seed_makes_ransac_reproducible() -> None:
    line_map: np.ndarray[tuple[int, int], np.dtype[np.uint8]] = np.zeros(
        (64, 96), dtype=np.uint8
    )
    cv2.line(line_map, (4, 12), (90, 12), 255, 1)
    cv2.line(line_map, (8, 55), (80, 20), 255, 1)

    first = extract_line_segments(
        line_map,
        config=_config(max_lines=4),
        rng=np.random.default_rng(123),
    ).segments
    second = extract_line_segments(
        line_map.copy(),
        config=_config(max_lines=4),
        rng=np.random.default_rng(123),
    ).segments

    np.testing.assert_array_equal(first, second)


def test_canonicalize_and_sort_are_deterministic() -> None:
    segments = np.asarray(
        [
            [0.8, 0.7, 0.2, 0.7],
            [0.5, 0.9, 0.5, 0.1],
            [0.1, 0.2, 0.9, 0.2],
        ],
        dtype=np.float32,
    )
    shuffled = segments[[2, 0, 1]]

    first = sort_and_pad_segments(segments, max_lines=5)
    second = sort_and_pad_segments(shuffled, max_lines=5)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(
        canonicalize_segment(np.asarray([4.0, 8.0, 1.0, 8.0])),
        np.asarray([1.0, 8.0, 4.0, 8.0], dtype=np.float32),
    )
    assert np.all(first[3:] == 0.0)


def test_empty_map_uses_all_zero_padding() -> None:
    result = extract_line_segments(
        np.zeros((32, 48), dtype=np.float32),
        config=_config(max_lines=3),
        rng=np.random.default_rng(0),
    )
    np.testing.assert_array_equal(result.segments, np.zeros((3, 4), dtype=np.float32))
    assert result.diagnostics.extracted_line_count == 0
