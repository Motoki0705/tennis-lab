"""Unit tests for peak decoding and original-image coordinate scaling."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.tasks.ball_detection.visualization.inference.peaks import (
    decode_frame_peaks,
    peaks_to_points,
)
from src.tasks.ball_detection.visualization.inference.rasters import (
    probability_raster,
    probability_rgba,
)


def _single_peak_heatmap(
    *,
    size_hw: tuple[int, int],
    cell_xy: tuple[int, int],
    value: float = 0.9,
) -> torch.Tensor:
    height, width = size_hw
    heatmap = torch.zeros((2, height, width), dtype=torch.float32)
    heatmap[:, cell_xy[1], cell_xy[0]] = value
    return heatmap


def test_peak_scaling_uses_original_size_minus_one() -> None:
    heatmap = _single_peak_heatmap(size_hw=(4, 8), cell_xy=(4, 2))
    peaks = decode_frame_peaks(
        heatmap,
        original_size=(1280, 720),
        threshold=0.5,
        nms_kernel=3,
        max_peaks=1,
        subpixel_refine=False,
    )
    assert len(peaks) == 2
    # Cell (4, 2) on an 8x4 grid is normalized (4/7, 2/3); scaling by
    # (width - 1, height - 1) reproduces the original pixel centre exactly.
    expected_x = 4.0 / 7.0 * 1279.0
    expected_y = 2.0 / 3.0 * 719.0
    assert peaks[0].points[0] == pytest.approx((expected_x, expected_y))
    assert peaks[0].scores[0] == pytest.approx(0.9)


def test_threshold_filters_and_keeps_multiple_peaks() -> None:
    heatmap = torch.zeros((1, 8, 8), dtype=torch.float32)
    heatmap[0, 1, 1] = 0.9
    heatmap[0, 6, 6] = 0.8
    heatmap[0, 3, 5] = 0.2
    peaks = decode_frame_peaks(
        heatmap,
        original_size=(100, 100),
        threshold=0.5,
        nms_kernel=3,
        max_peaks=4,
        subpixel_refine=False,
    )
    points = peaks[0].points
    assert len(points) == 2
    assert all(0.0 <= value <= 100.0 for point in points for value in point)
    # Both peaks are preserved; the sub-threshold cell contributes nothing.
    assert sorted(point[0] for point in points) == pytest.approx(
        [1 / 7 * 99, 6 / 7 * 99], rel=1e-5
    )


def test_subpixel_refinement_keeps_the_lattice_peak_for_isolated_cells() -> None:
    heatmap = _single_peak_heatmap(size_hw=(8, 8), cell_xy=(4, 4))
    peaks = decode_frame_peaks(
        heatmap,
        original_size=(64, 64),
        threshold=0.5,
        nms_kernel=3,
        max_peaks=1,
        subpixel_refine=True,
    )
    assert peaks[0].points[0] == pytest.approx((4.0 / 7.0 * 63.0, 4.0 / 7.0 * 63.0))


def test_empty_heatmap_yields_no_points() -> None:
    heatmap = torch.zeros((1, 4, 4), dtype=torch.float32)
    peaks = decode_frame_peaks(
        heatmap,
        original_size=(10, 10),
        threshold=0.5,
        nms_kernel=3,
        max_peaks=2,
        subpixel_refine=True,
    )
    assert peaks[0].points == ()
    assert peaks_to_points(peaks[0]) == []


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="shape"):
        decode_frame_peaks(
            torch.zeros((2, 3, 4, 5)),
            original_size=(10, 10),
            threshold=0.5,
            nms_kernel=3,
            max_peaks=1,
            subpixel_refine=False,
        )
    with pytest.raises(ValueError, match="positive"):
        decode_frame_peaks(
            torch.zeros((1, 4, 4)),
            original_size=(0, 10),
            threshold=0.5,
            nms_kernel=3,
            max_peaks=1,
            subpixel_refine=False,
        )


def test_points_payload_matches_frame_layer_contract() -> None:
    heatmap = _single_peak_heatmap(size_hw=(4, 4), cell_xy=(2, 1))
    peaks = decode_frame_peaks(
        heatmap,
        original_size=(80, 60),
        threshold=0.5,
        nms_kernel=3,
        max_peaks=1,
        subpixel_refine=False,
    )
    points = peaks_to_points(peaks[0])
    assert set(points[0]) == {"x", "y", "label", "score", "visible"}
    assert points[0]["label"] == "p1"
    assert points[0]["visible"] is True
    assert np.isfinite(points[0]["x"]) and np.isfinite(points[0]["y"])


def test_probability_raster_is_transparent_where_probability_is_zero() -> None:
    rgba = probability_rgba(np.asarray([[0.0, 1.0]], dtype=np.float32))
    assert rgba.shape == (1, 2, 4)
    assert rgba[0, 0, 3] == 0
    assert rgba[0, 1, 3] == 200
    raster = probability_raster(np.zeros((3, 3), dtype=np.float32))
    assert raster.name == "probability"
    data = raster.to_dict()["data"]
    assert isinstance(data, str)
    assert data.startswith("data:image/png;base64,")
    assert raster.to_dict()["legend"] == []


def test_probability_raster_is_deterministic_for_the_same_input() -> None:
    values = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    assert probability_raster(values).data_url == probability_raster(values).data_url


def test_non_finite_probability_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        probability_rgba(np.asarray([[np.nan]], dtype=np.float32))
