"""Unit tests for :mod:`src.utils.data.heatmaps`."""

from __future__ import annotations

import pytest
import torch

from src.utils.data.heatmaps import (
    generate_gaussian_heatmap,
    generate_gaussian_heatmaps,
    heatmaps_to_argmax,
    heatmaps_to_peaks,
    heatmaps_to_pixel_coords,
    heatmaps_to_soft_argmax,
)


class TestGenerateGaussianHeatmaps:
    def test_single_center_returns_squeezed_2d(self) -> None:
        hm = generate_gaussian_heatmaps((16, 16), (0.5, 0.5), sigma_ratio=0.1)
        assert hm.shape == (16, 16)

    def test_peak_is_at_center(self) -> None:
        hm = generate_gaussian_heatmaps((33, 33), (0.5, 0.5), sigma_ratio=0.05)
        peak = hm.argmax()
        row, col = divmod(int(peak), 33)
        assert (row, col) == (16, 16)
        assert hm.max().item() == pytest.approx(1.0, abs=1e-4)

    def test_batched_centers_shape(self) -> None:
        centers = torch.tensor([[0.2, 0.3], [0.7, 0.8], [0.5, 0.5]])
        hm = generate_gaussian_heatmaps((8, 12), centers, sigma_ratio=0.1)
        assert hm.shape == (3, 8, 12)

    def test_out_of_bounds_center_is_zeroed(self) -> None:
        hm = generate_gaussian_heatmaps((16, 16), (1.5, 0.5), sigma_ratio=0.1)
        assert torch.count_nonzero(hm) == 0

    def test_visibility_false_zeroes_map(self) -> None:
        centers = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        visibility = torch.tensor([True, False])
        hm = generate_gaussian_heatmaps(
            (16, 16), centers, sigma_ratio=0.1, visibility=visibility
        )
        assert hm[0].sum() > 0
        assert torch.count_nonzero(hm[1]) == 0

    def test_non_positive_sigma_raises(self) -> None:
        with pytest.raises(ValueError, match="sigma_ratio"):
            generate_gaussian_heatmaps((16, 16), (0.5, 0.5), sigma_ratio=0.0)

    def test_bad_center_shape_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\(\.\.\., 2\)"):
            generate_gaussian_heatmaps((16, 16), torch.zeros(3, 3), sigma_ratio=0.1)

    @pytest.mark.parametrize("size", [(0, 8), (8, -1), (8,)])
    def test_invalid_size_raises(self, size: tuple[int, ...]) -> None:
        with pytest.raises(ValueError):
            generate_gaussian_heatmaps(size, (0.5, 0.5), sigma_ratio=0.1)


class TestGenerateGaussianHeatmap:
    def test_visible_flag_false_zeroes(self) -> None:
        hm = generate_gaussian_heatmap((16, 16), (0.5, 0.5), 0.1, visible=False)
        assert torch.count_nonzero(hm) == 0

    def test_visible_true_has_peak(self) -> None:
        # Odd size so the normalized center 0.5 lands exactly on a grid point.
        hm = generate_gaussian_heatmap((17, 17), (0.5, 0.5), 0.1, visible=True)
        assert hm.max().item() == pytest.approx(1.0, abs=1e-4)


class TestHeatmapsToArgmax:
    def test_recovers_normalized_peak(self) -> None:
        hm = torch.zeros(9, 9)
        hm[0, 8] = 1.0  # bottom-left in (row=y, col=x) -> x=1.0, y=0.0
        coords, values = heatmaps_to_argmax(hm)
        assert torch.allclose(coords, torch.tensor([1.0, 0.0]))
        assert values.item() == 1.0

    def test_batched_leading_dims(self) -> None:
        hm = torch.rand(2, 3, 5, 7)
        coords, values = heatmaps_to_argmax(hm)
        assert coords.shape == (2, 3, 2)
        assert values.shape == (2, 3)

    def test_low_ndim_raises(self) -> None:
        with pytest.raises(ValueError):
            heatmaps_to_argmax(torch.zeros(5))


class TestHeatmapsToPixelCoords:
    def test_scales_to_pixel_grid(self) -> None:
        hm = torch.zeros(10, 20)
        hm[9, 19] = 1.0
        coords = heatmaps_to_pixel_coords(hm)
        assert torch.allclose(coords, torch.tensor([19.0, 9.0]))

    def test_custom_target_size(self) -> None:
        hm = torch.zeros(10, 10)
        hm[9, 9] = 1.0
        coords = heatmaps_to_pixel_coords(hm, height=100, width=200)
        assert torch.allclose(coords, torch.tensor([199.0, 99.0]))


class TestHeatmapsToSoftArgmax:
    def test_sharp_peak_approximates_argmax(self) -> None:
        hm = torch.full((11, 11), -10.0)
        hm[5, 5] = 10.0
        coords = heatmaps_to_soft_argmax(hm, temperature=0.5)
        assert torch.allclose(coords, torch.tensor([0.5, 0.5]), atol=1e-3)

    def test_is_differentiable(self) -> None:
        hm = torch.randn(8, 8, requires_grad=True)
        coords = heatmaps_to_soft_argmax(hm)
        coords.sum().backward()
        assert hm.grad is not None
        assert torch.isfinite(hm.grad).all()

    def test_non_positive_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            heatmaps_to_soft_argmax(torch.zeros(4, 4), temperature=0.0)

    def test_low_ndim_raises(self) -> None:
        with pytest.raises(ValueError):
            heatmaps_to_soft_argmax(torch.zeros(4))


class TestHeatmapsToPeaks:
    def test_extracts_single_peak(self) -> None:
        hm = torch.zeros(17, 17)
        hm[8, 8] = 1.0
        coords, values, valid = heatmaps_to_peaks(
            hm, threshold=0.5, nms_kernel=3, max_peaks=4
        )
        assert coords.shape == (4, 2)
        assert valid[0].item() is True
        assert values[0].item() == pytest.approx(1.0)
        assert torch.allclose(coords[0], torch.tensor([0.5, 0.5]), atol=1e-6)
        # Only one peak exceeds threshold.
        assert int(valid.sum()) == 1

    def test_threshold_filters_low_peaks(self) -> None:
        hm = torch.full((9, 9), 0.1)
        _, _, valid = heatmaps_to_peaks(
            hm, threshold=0.5, nms_kernel=3, max_peaks=4
        )
        assert int(valid.sum()) == 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"threshold": -1.0, "nms_kernel": 3, "max_peaks": 1},
            {"threshold": 0.0, "nms_kernel": 2, "max_peaks": 1},
            {"threshold": 0.0, "nms_kernel": 3, "max_peaks": 0},
        ],
    )
    def test_invalid_args_raise(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            heatmaps_to_peaks(torch.zeros(8, 8), **kwargs)
