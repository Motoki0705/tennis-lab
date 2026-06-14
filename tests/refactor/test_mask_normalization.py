"""Characterization tests: shared masking utilities reproduce task helpers.

Covers the consolidation of:
- plcs ``_masked_mean`` (binarize + clamp_min(1.0))
- blcs ``_masked_mean`` (broadcast + 1e-8 denom)
- plcs ``_to_frame_mask`` (flatten=False) and metrics ``_valid_from_human_mask``
  (flatten=True)
against the shared :func:`masked_mean` / :func:`normalize_padding_mask`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.utils.tensor_utils import masked_mean, normalize_padding_mask

GOLDENS = Path(__file__).parent / "goldens" / "masks.pt"


@pytest.fixture(scope="module")
def g() -> dict:
    return torch.load(GOLDENS, weights_only=True)


def test_masked_mean_plcs_convention(g: dict) -> None:
    out = masked_mean(g["values_bt"], g["mask_bt"], binarize=True, denom_min=1.0)
    assert torch.allclose(out, g["plcs_mm"], atol=1e-7)


def test_masked_mean_blcs_convention(g: dict) -> None:
    out = masked_mean(g["values_btj3"], g["mask_bt"], eps=1e-8)
    assert torch.allclose(out, g["blcs_mm"], atol=1e-7)


def test_masked_mean_none_mask_is_plain_mean() -> None:
    values = torch.randn(3, 4)
    assert torch.allclose(masked_mean(values, None), values.mean())


@pytest.mark.parametrize("key", ["mask_b", "mask_bt", "mask_bnt", "mask_bntj"])
def test_normalize_padding_mask_frame(g: dict, key: str) -> None:
    out = normalize_padding_mask(g[key], flatten=False)
    assert torch.equal(out, g[f"to_frame_{key}"])


@pytest.mark.parametrize("key", ["mask_b", "mask_bt", "mask_bnt", "mask_bntj"])
def test_normalize_padding_mask_flatten(g: dict, key: str) -> None:
    out = normalize_padding_mask(g[key], flatten=True)
    assert torch.equal(out, g[f"valid_{key}"])


def test_normalize_padding_mask_none() -> None:
    assert normalize_padding_mask(None) is None


def test_normalize_padding_mask_rejects_5d() -> None:
    with pytest.raises(ValueError):
        normalize_padding_mask(torch.ones(1, 1, 1, 1, 1))
