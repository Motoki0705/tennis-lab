"""Spatial pairing must separate order swaps, missing peaks and head assignment."""

import runpy
from pathlib import Path

import numpy as np
import pytest
import torch

from src.tasks.ball_refiner.refiner_2d.mean_anchors import candidate_mean_logits

SCRIPT = (Path(__file__).resolve().parents[5]
          / "knowledge/runs/run-i935-source-tail-audit-r27-20261001/audit_frames.py")


@pytest.fixture(scope="module")
def audit():
    return runpy.run_path(str(SCRIPT))


def test_pairing_detects_rank_swaps_but_does_not_match_invalid_or_distant_peaks(audit):
    a = np.array([[0., 0.], [100., 0.], [200., 0.], [0., 0.]])
    b = np.array([[101., 0.], [1., 0.], [400., 0.], [0., 0.]])
    valid = np.array([True, True, True, False])
    np.testing.assert_array_equal(audit["match_candidates"](a, b, valid, valid), [1, 0, -1, -1])


def test_pairing_maximizes_count_before_distance(audit):
    a = np.array([[0., 0.], [2., 0.]])
    b = np.array([[1., 0.], [-2., 0.]])
    np.testing.assert_array_equal(audit["match_candidates"](a, b, np.ones(2, bool), np.ones(2, bool), 2.), [1, 0])


def test_anchor_assignment_matches_model_score_and_coordinate_ties(audit):
    coords = np.array([[[.8, .3], [.4, .3], [.2, .5], [.9, .8]]], np.float32)
    scores = np.array([[.8, .8, .9, 1.]], np.float32)
    valid = np.array([[True, True, True, False]])
    order = audit["anchor_order"](coords, scores, valid)
    np.testing.assert_array_equal(order, [[2, 1, 0, 3]])
    features = torch.from_numpy(np.concatenate((coords, scores[..., None]), -1))
    logits = candidate_mean_logits(torch.zeros(1, 4, 2), features, torch.from_numpy(valid), count=3, max_offset_uv=.02)
    np.testing.assert_allclose(logits.sigmoid().numpy()[0, :3], coords[0, order[0, :3]], atol=1e-7)
