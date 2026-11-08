"""Small independent endpoint/color/mask cases for the research approximation."""

import runpy
from pathlib import Path

import numpy as np
import pytest

_module = runpy.run_path(str(Path(__file__).with_name("approximation.py")))
composite = _module["composite_decoded"]
parameters = _module["prefix_parameters"]


@pytest.mark.parametrize("mode", _module["MODES"])
def test_no_foreground_keeps_public_rgb_bits(mode):
    background = np.asarray([[[0.13, 0.27, 0.81], [0.2, 0.4, 0.6]]], dtype=np.float32)
    result = composite(
        np.zeros_like(background),
        [[[0.7], [1]]],
        [[[0.999], [0.999]]],
        np.ones_like(background),
        background,
        np.zeros((1, 2, 1), dtype=bool),
        mode=mode,
    )
    assert result.tobytes() == background.tobytes()


@pytest.mark.parametrize("mode", _module["MODES"])
def test_empty_prefix_keeps_foreground_without_decoder_bias(mode):
    foreground = np.asarray([[0.2, 0.5, 0.9]], dtype=np.float32)
    actual = composite(
        [[0.47, 0.33, 0.61]],
        [[1]],
        [[0.9999]],
        foreground,
        [[0.1, 0.1, 0.1]],
        np.ones((1, 1), bool),
        mode=mode,
    )
    np.testing.assert_array_equal(actual, foreground)


def test_full_alpha_closure_matches_original_even_if_accumulation_is_not_one():
    # Full accumulation .75 and foreground green. Only the relative closure must
    # return the original decoded background when prefix and full are identical.
    background = np.asarray([[0.2, 0.4, 0.8]], dtype=np.float32)
    actual = composite(
        background,
        [[0.25]],
        [[0.75]],
        [[0, 1, 0]],
        background,
        np.ones((1, 1), bool),
        mode="closed_normalized",
    )
    np.testing.assert_array_equal(actual, background)
    for mode in ("weighted", "normalized"):
        candidate = composite(
            background,
            [[0.25]],
            [[0.75]],
            [[0, 1, 0]],
            background,
            np.ones((1, 1), bool),
            mode=mode,
        )
        assert float(np.max(np.abs(candidate - background))) > 0.1


def test_half_linear_light_is_not_half_encoded_rgb():
    actual = composite(
        [[0, 0, 0]],
        [[0.625]],
        [[0.75]],
        [[1, 1, 1]],
        [[0.1, 0.2, 0.3]],
        np.ones((1, 1), bool),
        mode="closed_normalized",
    )
    np.testing.assert_allclose(
        actual, [[0.735356983, 0.735356983, 0.735356983]], atol=1e-7, rtol=0
    )


def test_decoder_scales_have_independent_hand_calculated_values():
    for mode, expected_scale, expected_weight in (
        ("weighted", 1, 0.25),
        ("normalized", 4, 0.25),
        ("closed_normalized", 3, 1 / 3),
    ):
        scale, weight = parameters([[0.75]], [[0.75]], mode)
        np.testing.assert_allclose(scale, [[expected_scale]], atol=1e-14, rtol=0)
        np.testing.assert_allclose(weight, [[expected_weight]], atol=1e-14, rtol=0)


def test_normalized_feature_is_invariant_to_uniform_prefix_attenuation():
    # The hypothetical primitive harmonic feature is .8; two opacities change
    # accumulated F, but normalized variants retain the same decoder input .6.
    values = []
    for alpha in (0.125, 0.5):
        scale, _ = parameters([[1 - alpha]], [[0.75]], "closed_normalized")
        values.append(float(scale[0, 0] * (alpha * 0.8)))
    np.testing.assert_allclose(values, [0.6, 0.6], atol=1e-14, rtol=0)


def test_tiny_prefix_is_explicitly_ignored_without_nan():
    actual = composite(
        [[1, 1, 1]],
        [[1 - 1e-8]],
        [[0.999]],
        [[0.12, 0.34, 0.56]],
        [[0.3, 0.3, 0.3]],
        np.ones((1, 1), bool),
        mode="closed_normalized",
    )
    np.testing.assert_array_equal(actual, np.asarray([[0.12, 0.34, 0.56]], np.float32))


@pytest.mark.parametrize(
    ("transmittance", "alpha"),
    [(-0.1, 1), (1.1, 1), (0, 0.9), (0.5, 0), (float("nan"), 1)],
)
def test_reject_inconsistent_opacity_instead_of_falling_back(transmittance, alpha):
    with pytest.raises(ValueError):
        parameters([[transmittance]], [[alpha]], "closed_normalized")


def test_reject_unknown_approximation():
    with pytest.raises(ValueError, match="Unknown"):
        parameters([[0.5]], [[1]], "ed_compare")


def test_reject_invalid_foreground_mask_and_rgb():
    for rgb, mask in (
        ([[float("nan"), 0, 0]], np.ones((1, 1), bool)),
        ([[1.1, 0, 0]], np.ones((1, 1), bool)),
        ([[0, 0, 0]], np.ones((1, 1), np.float32)),
    ):
        with pytest.raises(ValueError):
            composite(
                [[0.2, 0.3, 0.4]],
                [[0.5]],
                [[1]],
                rgb,
                [[0.1, 0.2, 0.3]],
                mask,
                mode="closed_normalized",
            )


def test_sphere_depth_matches_two_independently_known_surface_points():
    geometry = runpy.run_path(str(Path(__file__).with_name("geometry.py")))
    camera = {
        "width": 5,
        "height": 5,
        "intrinsics": {"matrix": [[4, 0, 2.5], [0, 4, 2.5], [0, 0, 1]]},
    }
    z, valid = geometry["sphere_field"](camera, 5, 128, 128, radius_m=np.sqrt(2))
    # On axis: near point is 5-sqrt(2). Ray x/z=.25 hits (1,0,4).
    assert valid[2, 2, 0] and valid[2, 3, 0]
    assert z[2, 2, 0] == pytest.approx(5 - np.sqrt(2), abs=2e-7)
    assert z[2, 3, 0] == pytest.approx(4, abs=2e-7)


def test_subpixel_ball_has_four_known_samples_on_common_ssaa2_grid():
    geometry = runpy.run_path(str(Path(__file__).with_name("geometry.py")))
    camera = {
        "width": 512,
        "height": 288,
        "intrinsics": {"matrix": [[360, 0, 256], [0, 360, 144], [0, 0, 1]]},
    }
    z, valid = geometry["sphere_field"](camera, 10, 128, 72)
    assert int(valid.sum()) == 4
    assert valid[143:145, 255:257].all()
    assert ((z[valid] > 9.96) & (z[valid] < 10)).all()
    assert (z[~valid] == 0).all()
