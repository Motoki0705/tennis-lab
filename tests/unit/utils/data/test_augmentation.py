"""Unit tests for :mod:`src.utils.data.augmentation`.

Covers the deterministic / validation-heavy public helpers. The stochastic
augmentations are exercised through their no-op early-returns, their error
paths, and (with a seeded generator) their shape/bounds invariants.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.utils.data.augmentation import (
    add_gaussian_noise,
    add_temporally_correlated_jitter,
    apply_burst_visibility_dropout,
    augment_keypoints,
    denormalize_tensor_images_imagenet,
    dilate_temporal_mask,
    inject_false_positive_observations,
    normalize_frames_imagenet,
    normalize_tensor_images_imagenet,
    parse_float_range,
    parse_int_range,
    random_visibility_dropout,
    scale_uv_with_visibility,
)


class TestParseFloatRange:
    def test_valid_range(self) -> None:
        assert parse_float_range([0.1, 0.9], "r") == (0.1, 0.9)

    @pytest.mark.parametrize("value", [[1.0], "ab", [3.0, 1.0], 5.0])
    def test_invalid_raises(self, value: object) -> None:
        with pytest.raises(ValueError, match="r"):
            parse_float_range(value, "r")


class TestParseIntRange:
    def test_valid_range(self) -> None:
        assert parse_int_range([1, 4], "r") == (1, 4)

    @pytest.mark.parametrize("value", [[1], [-1, 2], [3, 1]])
    def test_invalid_raises(self, value: object) -> None:
        with pytest.raises(ValueError, match="r"):
            parse_int_range(value, "r")


class TestImageNetNormalization:
    def test_round_trip(self) -> None:
        images = torch.rand(2, 3, 4, 4)
        normalized = normalize_tensor_images_imagenet(images)
        recovered = denormalize_tensor_images_imagenet(normalized)
        assert torch.allclose(recovered, images, atol=1e-6)

    def test_known_value(self) -> None:
        images = torch.full((3, 1, 1), 0.485)
        images[1] = 0.456
        images[2] = 0.406
        out = normalize_tensor_images_imagenet(images)
        # mean-subtracted channels become ~0 at the ImageNet mean.
        assert torch.allclose(out.squeeze(), torch.zeros(3), atol=1e-6)

    @pytest.mark.parametrize("shape", [(1, 4, 4), (2, 4, 4), (4,)])
    def test_bad_channel_shape_raises(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match="3, H, W"):
            normalize_tensor_images_imagenet(torch.zeros(shape))

    def test_numpy_frames(self) -> None:
        frames: list[np.ndarray] = [np.full((2, 2, 3), 0.5, dtype=np.float32)]
        out = normalize_frames_imagenet(frames)
        assert out[0].shape == (2, 2, 3)
        assert out[0].dtype == np.float32


class TestScaleUvWithVisibility:
    def test_identity_scale_keeps_points(self) -> None:
        uv = torch.tensor([[0.25, 0.75]])
        vis = torch.tensor([True])
        out_uv, out_vis = scale_uv_with_visibility(uv, vis, scale=1.0)
        assert torch.allclose(out_uv, uv)
        assert bool(out_vis[0])

    def test_scaling_out_of_bounds_drops_visibility(self) -> None:
        uv = torch.tensor([[0.95, 0.5]])
        vis = torch.tensor([True])
        _, out_vis = scale_uv_with_visibility(uv, vis, scale=3.0, center=0.5)
        assert not bool(out_vis[0])

    def test_non_positive_scale_raises(self) -> None:
        with pytest.raises(ValueError, match="scale"):
            scale_uv_with_visibility(torch.zeros(1, 2), torch.zeros(1), scale=0.0)


class TestAddGaussianNoise:
    def test_non_positive_std_is_noop(self) -> None:
        t = torch.arange(5.0)
        assert add_gaussian_noise(t, 0.0) is t

    def test_reproducible_with_generator(self, torch_generator: torch.Generator) -> None:
        t = torch.zeros(100)
        gen2 = torch.Generator().manual_seed(torch_generator.initial_seed())
        a = add_gaussian_noise(t, 0.5, generator=torch_generator)
        b = add_gaussian_noise(t, 0.5, generator=gen2)
        assert torch.allclose(a, b)
        assert a.shape == t.shape


class TestRandomVisibilityDropout:
    def test_zero_prob_is_noop(self) -> None:
        vis = torch.ones(10, dtype=torch.bool)
        assert random_visibility_dropout(vis, 0.0) is vis

    def test_full_prob_drops_everything(self) -> None:
        vis = torch.ones(50, dtype=torch.bool)
        out = random_visibility_dropout(vis, 1.0)
        assert int(out.sum()) == 0


class TestDilateTemporalMask:
    def test_zero_radius_returns_bool(self) -> None:
        mask = torch.tensor([0, 1, 0], dtype=torch.float32)
        out = dilate_temporal_mask(mask, 0)
        assert out.dtype == torch.bool
        assert out.tolist() == [False, True, False]

    def test_dilation_spreads_to_neighbors(self) -> None:
        mask = torch.tensor([False, False, True, False, False])
        out = dilate_temporal_mask(mask, radius=1)
        assert out.tolist() == [False, True, True, True, False]


class TestAddTemporallyCorrelatedJitter:
    def test_zero_std_is_noop(self) -> None:
        uv = torch.rand(3, 2)
        out = add_temporally_correlated_jitter(uv, jitter_std=0.0, drift_std=0.0)
        assert out is uv

    def test_invalid_drift_decay_raises(self) -> None:
        with pytest.raises(ValueError, match="drift_decay"):
            add_temporally_correlated_jitter(
                torch.rand(3, 2), jitter_std=0.1, drift_decay=1.0
            )

    def test_output_is_clamped(self, torch_generator: torch.Generator) -> None:
        uv = torch.rand(10, 2)
        out = add_temporally_correlated_jitter(
            uv, jitter_std=5.0, drift_std=5.0, generator=torch_generator
        )
        assert out.min() >= 0.0 and out.max() <= 1.0


class TestApplyBurstVisibilityDropout:
    def test_zero_prob_is_noop(self) -> None:
        vis = torch.ones(2, 8, dtype=torch.bool)
        out = apply_burst_visibility_dropout(
            vis, prob=0.0, min_len=1, max_len=2
        )
        assert out is vis

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"prob": 0.5, "min_len": 0, "max_len": 2},
            {"prob": 0.5, "min_len": 3, "max_len": 1},
        ],
    )
    def test_invalid_lengths_raise(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            apply_burst_visibility_dropout(torch.ones(1, 4), **kwargs)


class TestInjectFalsePositiveObservations:
    def test_no_prob_is_noop(self) -> None:
        uv = torch.rand(1, 4, 2)
        vis = torch.ones(1, 4)
        out_uv, out_vis = inject_false_positive_observations(uv, vis)
        assert out_uv is uv and out_vis is vis

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="visibility shape"):
            inject_false_positive_observations(
                torch.rand(1, 4, 2),
                torch.ones(1, 3),
                false_positive_prob=0.5,
            )


class TestAugmentKeypoints:
    def test_returns_same_shapes(self) -> None:
        kp = torch.rand(6, 2)
        vis = torch.ones(6, dtype=torch.bool)
        out_kp, out_vis = augment_keypoints(kp, vis, noise_std=0.1)
        assert out_kp.shape == kp.shape
        assert out_vis.shape == vis.shape
