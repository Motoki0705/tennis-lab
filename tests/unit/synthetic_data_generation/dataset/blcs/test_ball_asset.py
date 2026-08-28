"""Tests for the deterministic metric tennis-ball Gaussian asset."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from src.synthetic_data_generation.composition import gaussian_covariances
from src.synthetic_data_generation.dataset.blcs.ball_asset import (
    build_ball_gaussian_asset,
)


def test_ball_asset_is_a_deterministic_anisotropic_metric_surface(blcs_assets) -> None:
    first = build_ball_gaussian_asset(blcs_assets)
    second = build_ball_gaussian_asset(blcs_assets)
    settings = blcs_assets.settings

    assert first.gaussian_count == blcs_assets.ball.gaussian_count
    assert first.coordinates == blcs_assets.ball.coordinates
    assert first.appearance_model == "rgb"
    assert first.appearance_space == "linear_rgb"
    torch.testing.assert_close(first.means, second.means)
    torch.testing.assert_close(first.quaternions_wxyz, second.quaternions_wxyz)
    torch.testing.assert_close(first.features, second.features)
    torch.testing.assert_close(
        torch.linalg.vector_norm(first.means, dim=1),
        torch.full((first.gaussian_count,), settings.radius_m),
    )

    normals = torch.nn.functional.normalize(first.means, dim=1)
    covariances = gaussian_covariances(first)
    radial_variance = torch.einsum("ni,nij,nj->n", normals, covariances, normals)
    torch.testing.assert_close(
        radial_variance,
        torch.full_like(radial_variance, settings.radial_scale_m**2),
        rtol=2.0e-5,
        atol=1.0e-10,
    )
    torch.testing.assert_close(
        torch.diagonal(covariances, dim1=1, dim2=2).sum(dim=1),
        torch.full_like(
            radial_variance,
            2.0 * settings.tangential_scale_m**2 + settings.radial_scale_m**2,
        ),
        rtol=2.0e-5,
        atol=1.0e-10,
    )


def test_ball_asset_contains_felt_and_seam_materials(blcs_assets) -> None:
    asset = build_ball_gaussian_asset(blcs_assets)
    colors = torch.unique(asset.features, dim=0)
    seam_color = torch.tensor(
        blcs_assets.settings.seam_color_linear_rgb,
        dtype=asset.features.dtype,
    )
    seam_fraction = torch.all(torch.isclose(asset.features, seam_color), dim=1).float().mean()

    assert colors.shape == (2, 3)
    assert 0.01 < float(seam_fraction) < 0.2
    expected = torch.tensor(
        (
            blcs_assets.settings.base_color_linear_rgb,
            blcs_assets.settings.seam_color_linear_rgb,
        ),
        dtype=asset.features.dtype,
    )
    for color in expected:
        assert bool(torch.any(torch.all(torch.isclose(colors, color), dim=1)))
    torch.testing.assert_close(
        torch.sigmoid(asset.opacity_logits),
        torch.full_like(asset.opacity_logits, blcs_assets.settings.opacity),
    )


def test_ball_asset_contract_rejects_non_float32_public_payload(blcs_assets) -> None:
    with pytest.raises(ValueError, match="must use float32"):
        replace(
            blcs_assets,
            ball=replace(blcs_assets.ball, floating_dtype="float64"),
        )
