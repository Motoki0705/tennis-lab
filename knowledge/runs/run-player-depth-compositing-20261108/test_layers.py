"""Independent hand-calculated acceptance cases for the research reference."""

import runpy
from pathlib import Path

import numpy as np
import pytest

_reference = runpy.run_path(str(Path(__file__).with_name("layers.py")))
composite_prefix = _reference["composite_prefix"]
layer_reference = _reference["layer_reference"]


@pytest.mark.parametrize(
    ("depths", "alphas", "colors", "expected", "expected_t"),
    [
        ([2, 8], [1, 1], [[1, 0, 0], [0, 0, 1]], [1, 0, 0], 0),
        ([8], [1], [[0, 0, 1]], [0, 1, 0], 1),
        ([2, 8], [0.25, 1], [[1, 0, 0], [0, 0, 1]], [0.25, 0.75, 0], 0.75),
        ([6, 8], [0.25, 1], [[1, 0, 0], [0, 0, 1]], [0, 1, 0], 1),
        ([4], [1], [[1, 0, 0]], [0, 1, 0], 1),
        ([], [], [], [0, 1, 0], 1),
    ],
)
def test_layer_insertion_matches_hand_calculation(
    depths, alphas, colors, expected, expected_t
):
    result = layer_reference(depths, alphas, colors, 4, [0, 1, 0])
    np.testing.assert_allclose(result["composite_rgb"], expected, atol=1e-14, rtol=0)
    assert result["prefix_transmittance"] == expected_t


def test_identical_rgb_alpha_ed_cannot_determine_occlusion():
    a = layer_reference([2, 8], [0.5, 1], [[1, 0, 0], [1, 0, 0]], 4, [0, 1, 0])
    b = layer_reference([5], [1], [[1, 0, 0]], 4, [0, 1, 0])
    for key in ("background_rgb_premult", "background_alpha", "expected_z"):
        assert a[key] == b[key]
    assert a["expected_z"] == 5
    assert a["composite_rgb"] == [0.5, 0.5, 0]
    assert b["composite_rgb"] == [0, 1, 0]


def test_nonlinear_decode_of_accumulated_features_is_not_premultiplied_rgb():
    # D(f)=f², straight feature .8, alpha .25: actual color contribution is .16.
    # D(alpha*f)=.04; multiplying that result by alpha again gives .01.
    assert pytest.approx(0.04) == (0.25 * 0.8) ** 2
    assert pytest.approx(0.16) == 0.25 * 0.8**2
    assert pytest.approx(0.01) == 0.25 * (0.25 * 0.8) ** 2


def test_prefix_contract_uses_each_pixels_transmission_and_preserves_hdr():
    actual = composite_prefix(
        [[0.1, 0.2, 0], [0.5, 0, 0.25]],
        [[0.75], [0.25]],
        [[0, 2, 0], [1, 0.5, 0]],
    )
    np.testing.assert_allclose(
        actual, [[0.1, 1.7, 0], [0.75, 0.125, 0.25]], rtol=0, atol=1e-14
    )


@pytest.mark.parametrize(
    ("prefix", "transmission", "foreground"),
    [
        ([0, 0, 0], [-0.1], [0, 1, 0]),
        ([0, 0, 0], [1.1], [0, 1, 0]),
        ([0, np.nan, 0], [1], [0, 1, 0]),
        ([0, 0, 0], [1], [0, np.inf, 0]),
        ([[0, 0, 0]], [1], [[0, 1, 0]]),
    ],
)
def test_prefix_contract_rejects_invalid_inputs(prefix, transmission, foreground):
    with pytest.raises(ValueError):
        composite_prefix(prefix, transmission, foreground)
