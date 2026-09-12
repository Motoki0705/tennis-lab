"""Regression cases for the new regional-maximum decoder.

These exercise the decoder itself. Integration into the existing heatmaps
surface and court predictor is a separate, still-required change.
"""

from __future__ import annotations

import pytest
import torch

from src.utils.data.peaks import heatmaps_to_peaks


@pytest.mark.parametrize("background", [0.0, 0.05, 0.5, 0.9])
def test_constant_maps_never_invent_localized_keypoints(background: float) -> None:
    coords, scores, valid = heatmaps_to_peaks(
        torch.full((2, 14, 17, 17), background),
        threshold=0.05,
        nms_kernel=7,
        max_peaks=4,
    )
    assert coords.shape == (2, 14, 4, 2)
    assert not valid.any()
    assert not coords.any()
    assert not scores.any()


def _court_probabilities() -> torch.Tensor:
    maps = torch.full((14, 33, 33), 0.001)
    maps[:, 8, 8] = 0.99
    maps[:, 25, 25] = 0.07
    return maps


def test_threshold_rejects_reported_weak_secondary_peaks() -> None:
    coords, scores, valid = heatmaps_to_peaks(
        _court_probabilities(), threshold=0.5, nms_kernel=7, max_peaks=4
    )
    assert valid.sum().item() == 14
    assert valid[:, 0].all()
    torch.testing.assert_close(coords[:, 0], torch.tensor([0.25, 0.25]).expand(14, 2))
    torch.testing.assert_close(scores[:, 0], torch.full((14,), 0.99))
    assert not scores[:, 1:].any()


def test_lower_threshold_explicitly_retains_secondary_peaks() -> None:
    _, _, valid = heatmaps_to_peaks(
        _court_probabilities(), threshold=0.05, nms_kernel=7, max_peaks=4
    )
    assert valid.sum().item() == 28


def test_single_peak_capacity_never_returns_secondary_candidates() -> None:
    _, scores, valid = heatmaps_to_peaks(
        _court_probabilities(), threshold=0.05, nms_kernel=7, max_peaks=1
    )
    assert scores.shape == (14, 1)
    assert valid.sum().item() == 14
    torch.testing.assert_close(scores, torch.full((14, 1), 0.99))


def test_plateau_larger_than_nms_window_is_one_deterministic_peak() -> None:
    maps = torch.zeros(35, 35)
    maps[5:26, 5:26] = 0.9
    first = heatmaps_to_peaks(maps, threshold=0.5, nms_kernel=7, max_peaks=4)
    second = heatmaps_to_peaks(maps, threshold=0.5, nms_kernel=7, max_peaks=4)
    assert first[2].sum().item() == 1
    torch.testing.assert_close(first[0][0], torch.tensor([5 / 34, 5 / 34]))
    for actual, expected in zip(first, second, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_flat_region_touching_higher_value_is_not_a_regional_peak() -> None:
    maps = torch.zeros(35, 35)
    maps[5:26, 5:26] = 0.9
    maps[5, 5] = 1.0
    _, scores, valid = heatmaps_to_peaks(
        maps, threshold=0.5, nms_kernel=7, max_peaks=4
    )
    assert valid.sum().item() == 1
    assert scores[0].item() == 1.0


def test_equal_separate_peaks_obey_nms_without_losing_distant_peak() -> None:
    maps = torch.zeros(11, 11)
    maps[3, 3] = maps[3, 5] = maps[9, 9] = 0.9
    coords, _, valid = heatmaps_to_peaks(
        maps, threshold=0.5, nms_kernel=7, max_peaks=4
    )
    assert valid.sum().item() == 2
    torch.testing.assert_close(coords[:2], torch.tensor([[0.3, 0.3], [0.9, 0.9]]))


def test_capacity_padding_dtype_and_selected_score_gradient() -> None:
    maps = torch.tensor([[0.0, 0.9]], dtype=torch.float64, requires_grad=True)
    coords, scores, valid = heatmaps_to_peaks(
        maps, threshold=0.5, nms_kernel=3, max_peaks=6
    )
    assert coords.shape == (6, 2)
    assert valid.sum().item() == 1
    assert scores.dtype == torch.float64
    assert not coords[1:].any()
    assert not scores[1:].any()
    scores.sum().backward()
    torch.testing.assert_close(maps.grad, torch.tensor([[0.0, 1.0]], dtype=torch.float64))


def test_kernel_one_retains_distinct_neighboring_values() -> None:
    maps = torch.tensor([[0.0, 0.9, 0.8]])
    _, scores, valid = heatmaps_to_peaks(
        maps, threshold=0.5, nms_kernel=1, max_peaks=3
    )
    assert valid.sum().item() == 2
    torch.testing.assert_close(scores[:2], torch.tensor([0.9, 0.8]))


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_maps_fail_instead_of_hiding_invalid_predictions(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        heatmaps_to_peaks(
            torch.tensor([[0.0, value]]), threshold=0.5, nms_kernel=3, max_peaks=1
        )


@pytest.mark.parametrize("threshold", [-1.0, float("nan"), float("inf")])
def test_nonfinite_or_negative_threshold_fails(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        heatmaps_to_peaks(torch.zeros(3, 3), threshold=threshold, nms_kernel=3, max_peaks=1)


@pytest.mark.parametrize("kernel", [0, 2, -1])
def test_invalid_nms_fails(kernel: int) -> None:
    with pytest.raises(ValueError, match="nms_kernel"):
        heatmaps_to_peaks(torch.zeros(3, 3), threshold=0.5, nms_kernel=kernel, max_peaks=1)


def test_empty_spatial_axes_fail() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        heatmaps_to_peaks(torch.empty(2, 0, 3), threshold=0.5, nms_kernel=3, max_peaks=1)


def test_integer_maps_fail() -> None:
    with pytest.raises(TypeError, match="floating"):
        heatmaps_to_peaks(torch.zeros(3, 3, dtype=torch.long), threshold=0.5, nms_kernel=3, max_peaks=1)
