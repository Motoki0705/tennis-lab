"""Characterization tests: shared losses reproduce the pre-refactor task losses.

Goldens are produced by ``_generate_goldens.py`` from the original
``BallDetectionFocalLoss`` / court ``FocalBCEWithLogitsLoss`` implementations.
The shared :class:`FocalBCEWithLogitsLoss` must reproduce both bit-for-bit
(within floating-point tolerance).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.tasks.base.training.losses import FocalBCEWithLogitsLoss

GOLDENS = Path(__file__).parent / "goldens" / "focal_loss.pt"


@pytest.fixture(scope="module")
def golden() -> dict:
    return torch.load(GOLDENS, weights_only=True)


def test_shared_focal_matches_ball_detection(golden: dict) -> None:
    loss = FocalBCEWithLogitsLoss(gamma=2.0, validate_shape=True)
    out = loss(golden["logits"], golden["targets"])
    assert torch.allclose(out, golden["ball"], atol=1e-7)


def test_shared_focal_matches_court_detection(golden: dict) -> None:
    loss = FocalBCEWithLogitsLoss(gamma=2.0)
    out = loss(golden["logits"], golden["targets"])
    assert torch.allclose(out, golden["court"], atol=1e-7)


def test_negative_gamma_rejected() -> None:
    with pytest.raises(ValueError):
        FocalBCEWithLogitsLoss(gamma=-1.0)


def test_validate_shape_rejects_mismatch() -> None:
    loss = FocalBCEWithLogitsLoss(validate_shape=True)
    with pytest.raises(ValueError):
        loss(torch.zeros(2, 3), torch.zeros(2, 4))
