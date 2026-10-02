"""Calibration changes covariance only and requires explicit artifact identity."""

import json

import pytest
import torch

from src.tasks.ball_refiner.refiner_2d.calibration import (
    CovarianceCalibration,
    load_covariance_calibration,
)
from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D
from src.utils.checksum import dual_sha256


def test_full_covariance_scales_without_changing_locations_weights_or_presence():
    prediction = BallGMM2D(
        means=torch.tensor([[[[.2, .3], [.8, .7]]]], dtype=torch.float64),
        scale_tril=torch.tensor([[[[[.02, 0], [.01, .03]], [[.04, 0], [-.01, .05]]]]], dtype=torch.float64),
        mixture_logits=torch.tensor([[[.4, -.2]]], dtype=torch.float64),
        presence_logits=torch.tensor([[1.2]], dtype=torch.float64),
    )
    adjusted = CovarianceCalibration(9, "a" * 64).apply(prediction)
    torch.testing.assert_close(adjusted.covariance, 9 * prediction.covariance)
    for name in ("means", "mixture_logits", "presence_logits", "weights", "presence_probability"):
        assert torch.equal(getattr(adjusted, name), getattr(prediction, name))
    assert not torch.equal(prediction.scale_tril, adjusted.scale_tril)


@pytest.mark.parametrize("multiplier", [0, -1, float("nan"), float("inf"), True])
def test_invalid_scale_is_never_replaced_with_identity(multiplier):
    with pytest.raises(ValueError, match="multiplier"):
        CovarianceCalibration(multiplier, "a" * 64)


def test_loader_requires_artifact_and_checkpoint_hash_and_complete_schema(tmp_path):
    path = tmp_path / "calibration.json"
    raw = {"schema": "ball_refiner_2d.covariance_calibration.v1",
           "covariance_multiplier": 2., "checkpoint_sha256": "a" * 64,
           "provenance": {"fit_groups": ["clip1", "clip2"]}}
    path.write_text(json.dumps(raw))
    digest = dual_sha256(path)
    assert load_covariance_calibration(path, expected_sha256=digest, checkpoint_sha256="a" * 64).covariance_multiplier == 2
    with pytest.raises(ValueError, match="checkpoint"):
        load_covariance_calibration(path, expected_sha256=digest, checkpoint_sha256="b" * 64)
    path.write_text(json.dumps({**raw, "covariance_multiplier": 3}))
    with pytest.raises(ValueError, match="SHA256"):
        load_covariance_calibration(path, expected_sha256=digest, checkpoint_sha256="a" * 64)
    path.write_text(json.dumps({k: v for k, v in raw.items() if k != "covariance_multiplier"}))
    with pytest.raises(ValueError, match="incomplete"):
        load_covariance_calibration(path, expected_sha256=dual_sha256(path), checkpoint_sha256="a" * 64)
