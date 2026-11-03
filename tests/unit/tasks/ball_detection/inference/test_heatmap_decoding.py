"""Analytic centres, asymmetric axes and explicit rejected-fit contracts."""

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from src.tasks.ball_detection.inference.heatmap_decoding import (
    METHODS,
    FitStatus,
    HeatmapDecodingConfig,
    correct_coordinates,
    decode_heatmaps,
    decode_native_sequence,
    refine_peak_cells,
)
from src.utils.data.heatmaps import heatmaps_to_argmax, refine_peaks_log_parabolic
from tests.support.ball_heatmap_decoding import compose_heatmap_decoding


@pytest.fixture(scope="module")
def decoding_config() -> HeatmapDecodingConfig:
    return cast(HeatmapDecodingConfig, compose_heatmap_decoding())


def _map(centre=(7.23, 5.36), *, rotated=False, sigmoid=False, background=0., edge=False):
    yy, xx = np.mgrid[:13, :19].astype(np.float64)
    dx, dy = xx - centre[0], yy - centre[1]
    energy = (dx / 1.35) ** 2 + (dy / .85) ** 2
    if rotated:
        energy = ((.8 * dx + .6 * dy) / 1.65) ** 2 + ((-.6 * dx + .8 * dy) / .75) ** 2
    values = 1 / (1 + np.exp(np.clip(.5 * energy - 5., -50, 50))) if sigmoid else background + .8 * np.exp(-.5 * energy)
    return torch.tensor(values, dtype=torch.float32)


@pytest.mark.parametrize("method,patch,sigmoid,rotated,tolerance", [
    ("legacy", 5, False, False, 2e-5),
    ("axis_logit", 3, True, False, 5e-5),
    ("quadratic_log", 3, False, True, 2e-5),
    ("quadratic_log", 5, False, True, 2e-5),
    ("quadratic_logit", 3, True, True, 5e-5),
    ("quadratic_logit", 5, True, True, 5e-5),
    ("centroid", 5, False, False, .15),
    ("softargmax", 5, False, False, .15),
    ("gaussian", 5, False, True, 2e-3),
])
def test_known_centres_include_rotation_anisotropy_and_saturation(method, patch, sigmoid, rotated, tolerance, decoding_config):
    centre = np.array([7.23, 5.36])
    background = .07 if method in ("centroid", "softargmax", "gaussian") else 0.
    image = _map(tuple(centre), rotated=rotated, sigmoid=sigmoid, background=background)
    result = decode_heatmaps(image, replace(decoding_config, method=method, patch_size=patch), subpixel_refine=True)
    np.testing.assert_allclose(result.coords.numpy() * [18, 12], centre, atol=tolerance, rtol=0)
    assert result.confidence.item() == image.max().item()
    assert result.fit_status.item() in (FitStatus.LEGACY, FitStatus.REFINED)


@pytest.mark.parametrize("method,centre,tolerance", [
    ("quadratic_log", (.23, .36), 2e-5),
    ("quadratic_logit", (17.73, 11.67), 1e-4),
    ("gaussian", (.23, .36), 2e-3),
    ("centroid", (.23, .36), .65),
    ("softargmax", (17.73, 11.67), .65),
    ("axis_logit", (.23, 5.36), .24),
])
def test_border_uses_only_observed_patch_cells(method, centre, tolerance, decoding_config):
    image = _map(centre, sigmoid="logit" in method, background=.07 if method == "gaussian" else 0.)
    result = decode_heatmaps(image, replace(decoding_config, method=method), subpixel_refine=True)
    np.testing.assert_allclose(result.coords.numpy() * [18, 12], centre, atol=tolerance, rtol=0)
    if method == "axis_logit":
        assert result.fit_status.item() == FitStatus.PARTIAL_AXIS


@pytest.mark.parametrize("method", METHODS[1:])
def test_unidentifiable_flat_patch_reports_reason_and_keeps_lattice(method, decoding_config):
    result = decode_heatmaps(torch.full((7, 11), .5), replace(decoding_config, method=method), subpixel_refine=True)
    assert result.fit_status.item() == FitStatus.DEGENERATE
    torch.testing.assert_close(result.coords, torch.zeros(2), rtol=0, atol=0)


def test_default_matches_main_decoder_arrays_and_preserves_camera_time_axes(decoding_config):
    frozen = json.loads((Path(__file__).parent / "fixtures/main_decode.json").read_text())
    main_result = decode_heatmaps(torch.tensor(frozen["heatmaps"], dtype=torch.float32), decoding_config, subpixel_refine=True)
    assert torch.equal(main_result.coords, torch.tensor(frozen["coords"], dtype=torch.float32))
    assert torch.equal(main_result.confidence, torch.tensor(frozen["confidence"], dtype=torch.float32))
    # Two cameras, two unequal frame centres; shape alone cannot detect a swap.
    images = torch.stack([_map(c) for c in [(7.23, 5.36), (3.4, 8.12), (12.17, 2.34), (.23, .36)]]).reshape(2, 2, 13, 19)
    for subpixel in (False, True):
        peak, score = heatmaps_to_argmax(images)
        expected = refine_peaks_log_parabolic(images, peak) if subpixel else peak
        result = decode_heatmaps(images, decoding_config, subpixel_refine=subpixel)
        assert torch.equal(result.coords, expected)
        assert torch.equal(result.confidence, score)
    raw = images.reshape(4, 13, 19).numpy()
    raw.flags.writeable = False
    uv, scores, _ = decode_native_sequence(raw, decoding_config, subpixel_refine=True)
    np.testing.assert_array_equal(uv, expected.reshape(4, 2).numpy())
    np.testing.assert_array_equal(scores, score.reshape(4).numpy())


def test_coordinate_hypotheses_have_known_pixel_displacements(decoding_config):
    uv = torch.tensor([[0., 1.], [.25, .75]], dtype=torch.float64)
    config = replace(decoding_config, coordinate_correction="stride")
    pixels = correct_coordinates(uv, (72, 128), config) * torch.tensor([1919., 1079.])
    # Stride-four half-pixel centres at 1.5 and 285.5 in a 512x288 input.
    np.testing.assert_allclose(pixels[0], [1919 * 1.5 / 511, 1079 * 285.5 / 287])
    affine = replace(config, coordinate_correction="affine", scale_xy=(.8, 1.2), offset_xy=(.1, -.1))
    torch.testing.assert_close(correct_coordinates(uv, (72, 128), affine), torch.tensor([[.1, 1.], [.3, .8]], dtype=torch.float64))


@pytest.mark.parametrize("change", [
    {"method": "typo"}, {"patch_size": 4}, {"temperature": 0.},
    {"method": "gaussian", "patch_size": 3}, {"offset_xy": (.01, 0.)},
])
def test_invalid_decoding_settings_are_rejected(change, decoding_config):
    with pytest.raises(ValueError):
        replace(decoding_config, **change)


def test_nan_input_and_disabled_refinement_are_rejected(decoding_config):
    with pytest.raises(ValueError, match="finite"):
        decode_heatmaps(torch.full((3, 4), float("nan")), decoding_config, subpixel_refine=True)
    with pytest.raises(ValueError, match="subpixel_refine"):
        decode_heatmaps(_map(), replace(decoding_config, method="centroid"), subpixel_refine=False)


def test_isotropic_gaussian_and_exactly_saturated_plateau(decoding_config):
    yy, xx = np.mgrid[:13, :19].astype(np.float64)
    energy = (xx - 7.23) ** 2 + (yy - 5.36) ** 2
    image = torch.tensor(.8 * np.exp(-energy / 2), dtype=torch.float32)
    result = decode_heatmaps(image, replace(decoding_config, method="quadratic_log"), subpixel_refine=True)
    np.testing.assert_allclose(result.coords.numpy() * [18, 12], [7.23, 5.36], atol=2e-5)
    saturated = torch.tensor(1 / (1 + np.exp(np.clip(energy - 20, -50, 50))), dtype=torch.float32)
    assert saturated.max() == 1.
    # Saturated float32 loses centre information. This bounds the damage rather
    # than pretending that a logit transform can recover the lost bits.
    clipped = decode_heatmaps(saturated, replace(decoding_config, method="quadratic_logit"), subpixel_refine=True)
    assert torch.isfinite(clipped.coords).all()
    assert clipped.fit_status.item() != FitStatus.LEGACY


def test_explicit_cells_refine_two_distinct_peaks_without_argmax_or_padding_bias(decoding_config):
    yy, xx = np.mgrid[:13, :23]
    centers = np.array([[4.21, 3.16], [17.28, 8.32]])
    maps = [1 / (1 + np.exp(np.clip(((xx-x)/1.4)**2+((yy-y)/.9)**2-3, -50, 50))) for x,y in centers]
    image = np.maximum(*maps).astype(np.float32)
    original = image.copy()
    cfg = replace(decoding_config, method="quadratic_logit", patch_size=3,
        coordinate_correction="affine", scale_xy=(1.01,.99), offset_xy=(.01,-.02))
    uv, codes = refine_peak_cells(image, np.array([[4,3],[17,8],[0,0]],np.int64), np.array([True,True,False]), cfg)
    expected = centers / [22,12] * [1.01,.99] + [.01,-.02]
    np.testing.assert_allclose(uv[:2],expected,atol=1e-6,rtol=0)
    np.testing.assert_array_equal(uv[2],[0,0])
    np.testing.assert_array_equal(codes[:2],[FitStatus.REFINED]*2)
    np.testing.assert_array_equal(image,original)


@pytest.mark.parametrize("flat,code",[(True,FitStatus.DEGENERATE),(False,FitStatus.INSUFFICIENT_PATCH)])
def test_explicit_cells_keep_lattice_and_rejection_status(flat,code,decoding_config):
    image = np.full((5,7),.5,np.float32) if flat else _map((.1,.2)).numpy()
    uv,codes = refine_peak_cells(image,np.array([[0,0]],np.int64),np.array([True]),
        replace(decoding_config,method="quadratic_logit",patch_size=3))
    np.testing.assert_array_equal(uv,[[0,0]])
    assert codes[0] == code


@pytest.mark.parametrize("failure",["axes","bounds","legacy"])
def test_explicit_cells_reject_invalid_coordinates_and_implicit_legacy(failure,decoding_config):
    cells = np.array([[4,3]],np.int64)
    if failure=="axes":
        cells=cells[0]
    elif failure=="bounds":
        cells[0,0]=19
    config=decoding_config if failure=="legacy" else replace(decoding_config,method="quadratic_logit")
    with pytest.raises(ValueError):
        refine_peak_cells(_map().numpy(),cells,np.array([True]),config)
