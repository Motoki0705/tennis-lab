"""Regression tests for plateau-safe heatmap peak extraction."""

from __future__ import annotations

import pytest
import torch

from src.utils.data.heatmaps import heatmaps_to_peaks


def test_uniform_map_at_threshold_emits_no_peaks() -> None:
    heatmap = torch.full((9, 9), 0.05)

    _, values, valid = heatmaps_to_peaks(
        heatmap,
        threshold=0.05,
        nms_kernel=7,
        max_peaks=4,
    )

    assert valid.tolist() == [False, False, False, False]
    torch.testing.assert_close(values, torch.zeros(4))


def test_equal_valued_plateau_is_reduced_to_one_deterministic_peak() -> None:
    heatmap = torch.zeros(9, 9)
    heatmap[3:5, 4:6] = 0.9

    coords, values, valid = heatmaps_to_peaks(
        heatmap,
        threshold=0.5,
        nms_kernel=3,
        max_peaks=4,
    )

    assert int(valid.sum()) == 1
    assert values[0].item() == pytest.approx(0.9)
    torch.testing.assert_close(coords[0], torch.tensor([5.0 / 8.0, 4.0 / 8.0]))


def test_spatially_separated_peaks_remain_available_for_multi_peak_tasks() -> None:
    heatmap = torch.zeros(9, 9)
    heatmap[2, 2] = 0.9
    heatmap[6, 6] = 0.8

    coords, values, valid = heatmaps_to_peaks(
        heatmap,
        threshold=0.5,
        nms_kernel=3,
        max_peaks=4,
    )

    assert valid.tolist() == [True, True, False, False]
    torch.testing.assert_close(values[:2], torch.tensor([0.9, 0.8]))
    torch.testing.assert_close(
        coords[:2],
        torch.tensor([[2.0 / 8.0, 2.0 / 8.0], [6.0 / 8.0, 6.0 / 8.0]]),
    )


def test_non_finite_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        heatmaps_to_peaks(
            torch.zeros(3, 3),
            threshold=float("nan"),
            nms_kernel=3,
            max_peaks=1,
        )
