"""Unit tests for :mod:`src.utils.data.heatmaps`."""

from __future__ import annotations

from typing import cast

import pytest
import torch

from src.utils.data.heatmaps import (
    generate_gaussian_heatmap,
    generate_gaussian_heatmaps,
    heatmaps_to_argmax,
    heatmaps_to_peaks,
    heatmaps_to_pixel_coords,
    heatmaps_to_soft_argmax,
    refine_peaks_log_parabolic,
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

    def test_max_reduction_preserves_multiple_visible_peaks_per_channel(self) -> None:
        centers = torch.tensor([[[0.25, 0.25], [0.75, 0.75]]])
        heatmaps = generate_gaussian_heatmaps(
            (17, 17),
            centers,
            sigma_ratio=0.02,
            visibility=torch.tensor([[True, True]]),
            point_reduction="max",
        )

        assert heatmaps.shape == (1, 17, 17)
        assert heatmaps[0, 4, 4].item() == pytest.approx(1.0)
        assert heatmaps[0, 12, 12].item() == pytest.approx(1.0)

    def test_max_reduction_requires_explicit_point_axis(self) -> None:
        with pytest.raises(ValueError, match="explicit point axis"):
            generate_gaussian_heatmaps(
                (17, 17),
                (0.5, 0.5),
                sigma_ratio=0.02,
                point_reduction="max",
            )

    def test_non_positive_sigma_raises(self) -> None:
        with pytest.raises(ValueError, match="sigma_ratio"):
            generate_gaussian_heatmaps((16, 16), (0.5, 0.5), sigma_ratio=0.0)

    def test_bad_center_shape_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\(\.\.\., 2\)"):
            generate_gaussian_heatmaps((16, 16), torch.zeros(3, 3), sigma_ratio=0.1)

    @pytest.mark.parametrize("size", [(0, 8), (8, -1), (8,)])
    def test_invalid_size_raises(self, size: tuple[int, ...]) -> None:
        with pytest.raises(ValueError):
            generate_gaussian_heatmaps(
                cast("tuple[int, int]", size), (0.5, 0.5), sigma_ratio=0.1
            )


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

    def test_no_mask_retains_exact_normalized_contract(self) -> None:
        heatmaps = torch.tensor(
            [[[0.0, 1.0, -1.0], [2.0, -2.0, 0.5]]],
            dtype=torch.float64,
        )
        flat_probs = torch.softmax(heatmaps.flatten(-2) / 0.7, dim=-1).reshape_as(
            heatmaps
        )
        expected = torch.stack(
            (
                (
                    flat_probs.sum(dim=-2)
                    * torch.linspace(0.0, 1.0, 3, dtype=torch.float64)
                ).sum(dim=-1),
                (
                    flat_probs.sum(dim=-1)
                    * torch.linspace(0.0, 1.0, 2, dtype=torch.float64)
                ).sum(dim=-1),
            ),
            dim=-1,
        )

        actual = heatmaps_to_soft_argmax(heatmaps, temperature=0.7)

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_mask_excludes_adversarial_padding_value_and_gradient(self) -> None:
        heatmaps = torch.full((2, 1, 4, 5), -20.0, requires_grad=True)
        with torch.no_grad():
            heatmaps[0, 0, 1, 2] = 20.0
            heatmaps[0, 0, 3, 4] = 100.0
            heatmaps[1, 0, 1, 2] = 20.0
            heatmaps[1, 0, 2:, 3:] = 100.0
        valid_mask = torch.zeros_like(heatmaps, dtype=torch.bool)
        valid_mask[0, :, :3, :4] = True
        valid_mask[1, :, :2, :3] = True

        normalized = heatmaps_to_soft_argmax(
            heatmaps,
            temperature=0.5,
            valid_mask=valid_mask,
        )
        pixel_scale = heatmaps.new_tensor([4.0, 3.0])
        points_px = normalized * pixel_scale
        points_px.sum().backward()

        expected = heatmaps.new_tensor([2.0, 1.0])
        torch.testing.assert_close(points_px[:, 0], expected.expand(2, 2))
        assert heatmaps.grad is not None
        assert torch.count_nonzero(heatmaps.grad[~valid_mask]) == 0
        assert torch.isfinite(heatmaps.grad).all()

    def test_masked_uniform_distribution_has_known_expectation(self) -> None:
        heatmaps = torch.zeros((1, 1, 3, 4), dtype=torch.float64)
        valid_mask = torch.zeros_like(heatmaps, dtype=torch.bool)
        valid_mask[:, :, :2, :2] = True

        coords = heatmaps_to_soft_argmax(heatmaps, valid_mask=valid_mask)

        torch.testing.assert_close(
            coords,
            torch.tensor([[[1.0 / 6.0, 1.0 / 4.0]]], dtype=torch.float64),
        )

    @pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_temperature_raises(self, temperature: float) -> None:
        with pytest.raises(ValueError, match="temperature"):
            heatmaps_to_soft_argmax(torch.zeros(4, 4), temperature=temperature)

    def test_invalid_masks_raise(self) -> None:
        heatmaps = torch.zeros((2, 3, 4, 5))
        with pytest.raises(ValueError, match="same shape"):
            heatmaps_to_soft_argmax(
                heatmaps,
                valid_mask=torch.ones((2, 1, 4, 5), dtype=torch.bool),
            )
        with pytest.raises(TypeError, match="boolean"):
            heatmaps_to_soft_argmax(heatmaps, valid_mask=torch.ones_like(heatmaps))
        all_invalid = torch.ones_like(heatmaps, dtype=torch.bool)
        all_invalid[1, 2] = False
        with pytest.raises(ValueError, match="at least one valid"):
            heatmaps_to_soft_argmax(heatmaps, valid_mask=all_invalid)

    def test_mask_device_must_match(self) -> None:
        heatmaps = torch.zeros((2, 3))
        meta_mask = torch.ones((2, 3), dtype=torch.bool, device="meta")
        with pytest.raises(ValueError, match="same device"):
            heatmaps_to_soft_argmax(heatmaps, valid_mask=meta_mask)

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


class TestRefinePeaksLogParabolic:
    def test_recovers_fractional_gaussian_center_exactly(self) -> None:
        # Off-lattice center: argmax alone quantizes, refinement must recover it.
        center = torch.tensor([0.4130, 0.5020])
        hm = generate_gaussian_heatmaps((72, 128), center, sigma_ratio=0.012)
        coords, _ = heatmaps_to_argmax(hm)
        assert not torch.allclose(coords, center, atol=1e-3)
        refined = refine_peaks_log_parabolic(hm, coords)
        assert torch.allclose(refined, center, atol=1e-4)

    def test_batched_and_peak_axis_shapes(self) -> None:
        centers = torch.tensor([[0.31, 0.62], [0.77, 0.18]])
        hm = generate_gaussian_heatmaps((36, 64), centers, sigma_ratio=0.02)
        coords, _, _ = heatmaps_to_peaks(hm, threshold=0.5, nms_kernel=3, max_peaks=2)
        refined = refine_peaks_log_parabolic(hm, coords)
        assert refined.shape == coords.shape
        for index in range(2):
            assert torch.allclose(refined[index, 0], centers[index], atol=1e-4)

    def test_lattice_center_is_unchanged(self) -> None:
        # Center on a lattice point: refinement must not move it.
        hm = generate_gaussian_heatmaps((17, 17), (0.5, 0.5), sigma_ratio=0.05)
        coords, _ = heatmaps_to_argmax(hm)
        refined = refine_peaks_log_parabolic(hm, coords)
        assert torch.allclose(refined, coords, atol=1e-6)

    def test_border_peak_keeps_argmax_coordinate(self) -> None:
        hm = torch.zeros(9, 9)
        hm[0, 8] = 1.0
        coords, _ = heatmaps_to_argmax(hm)
        refined = refine_peaks_log_parabolic(hm, coords)
        assert torch.allclose(refined, coords)

    def test_flat_map_keeps_argmax_coordinate(self) -> None:
        hm = torch.full((9, 9), 0.5)
        coords, _ = heatmaps_to_argmax(hm)
        refined = refine_peaks_log_parabolic(hm, coords)
        assert torch.allclose(refined, coords)

    def test_offset_clamped_to_half_cell(self) -> None:
        hm = torch.zeros(9, 9)
        hm[4, 4] = 0.9
        hm[4, 5] = 0.89999
        coords, _ = heatmaps_to_argmax(hm)
        refined = refine_peaks_log_parabolic(hm, coords)
        assert (refined - coords).abs().max() <= 0.5 / 8 + 1e-6

    def test_mismatched_leading_shape_raises(self) -> None:
        hm = torch.rand(2, 8, 8)
        with pytest.raises(ValueError, match="leading dimensions"):
            refine_peaks_log_parabolic(hm, torch.rand(3, 2))

    def test_bad_last_dim_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\(\.\.\., 2\)"):
            refine_peaks_log_parabolic(torch.rand(8, 8), torch.rand(3))
