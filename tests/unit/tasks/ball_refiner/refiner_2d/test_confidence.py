"""Consumer uncertainty includes all mixture modes and retains explicit rejection."""

import math
import subprocess
import sys

import numpy as np
import pytest
import torch

from src.tasks.ball_refiner.refiner_2d.confidence import (
    PointConfidenceRule,
    point_confidence,
)
from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D


def mixture(means: list[list[float]], weights: list[float]) -> BallGMM2D:
    n = len(means)
    return BallGMM2D(torch.tensor([[means]]), torch.eye(2).repeat(1, 1, n, 1, 1) * .01,
                    torch.tensor([[weights]]).log(), torch.tensor([[math.log(9.)]]))


def test_single_gaussian_area_and_anisotropic_source_units() -> None:
    prediction = mixture([[.2, .3]], [1.])
    point, presence, area = point_confidence(prediction, (101, 201))
    np.testing.assert_allclose(point, [[[20, 60]]])
    np.testing.assert_allclose(presence, [[.9]])
    np.testing.assert_allclose(area, [[40 * math.pi]])


def test_separated_minor_mode_increases_uncertainty_without_averaging_point() -> None:
    near = mixture([[.2, .3], [.2, .3]], [.9, .1])
    far = mixture([[.2, .3], [.9, .9]], [.9, .1])
    p, _, a = point_confidence(near, (101, 101))
    q, _, b = point_confidence(far, (101, 101))
    np.testing.assert_array_equal(p, q)
    assert float(b[0, 0]) > 20 * float(a[0, 0])
    rule = PointConfidenceRule(.5, 2 * float(a[0, 0]))
    assert rule.rejection_codes(np.ones((1, 1)), b).item() == 2


def test_rule_boundaries_and_both_rejection_reasons() -> None:
    rule = PointConfidenceRule(.9, 30000.)
    result = rule.rejection_codes(np.array([.9, .8, .99, .1]), np.array([30000., 1., 30001., 30001.]))
    np.testing.assert_array_equal(result, [0, 1, 2, 3])
    with pytest.raises(ValueError, match="finite"):
        rule.rejection_codes(np.array([np.nan]), np.array([1.]))


def test_rule_is_frame_local_and_covariance_scaling_changes_uncertainty() -> None:
    original = mixture([[.2, .3]], [1.])
    scaled = BallGMM2D(original.means, original.scale_tril * 2, original.mixture_logits, original.presence_logits)
    _, _, area = point_confidence(original, (101, 101))
    _, _, large = point_confidence(scaled, (101, 101))
    np.testing.assert_allclose(large, area * 4)
    repeated = BallGMM2D(original.means.repeat(1, 7, 1, 1), original.scale_tril.repeat(1, 7, 1, 1, 1),
                        original.mixture_logits.repeat(1, 7, 1), original.presence_logits.repeat(1, 7))
    np.testing.assert_array_equal(point_confidence(repeated, (101, 101))[2], np.repeat(area, 7, axis=1))


def test_fresh_task_import_does_not_eagerly_import_scene_composition() -> None:
    subprocess.run([sys.executable, "-c", "from src.tasks.ball_refiner.pipeline_options import E9_ANCHORED_S42; "
                    "from src.tennis_scene.pipeline import TennisSceneOrchestrator; "
                    "assert E9_ANCHORED_S42.checkpoint_sha256 and TennisSceneOrchestrator"], check=True)
