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


@pytest.fixture(scope="module")
def noise():
    import sys
    from unittest.mock import patch

    with patch.object(sys, "path", [str(SCRIPT.parent), *sys.path]):
        return runpy.run_path(str(SCRIPT.with_name("audit_noise.py")))


def test_bootstrap_blocks_keep_original_time_adjacency_without_wraparound(noise):
    indices = noise["block_indices"](11, 4, 100, np.random.default_rng(1729))
    assert indices.shape == (100, 11)
    assert indices.min() >= 0 and indices.max() < 11
    np.testing.assert_equal(np.diff(indices[:, :8].reshape(100, 2, 4), axis=-1), 1)
    np.testing.assert_equal(np.diff(indices[:, 8:], axis=-1), 1)


def test_paired_bootstrap_retains_missing_gt_and_known_constant_difference(noise):
    cached: np.ndarray = np.arange(11, dtype=float)
    cached[[0, 3]] = np.nan
    indices = noise["block_indices"](11, 4, 100, np.random.default_rng(1729))
    np.testing.assert_allclose(noise["p90_difference"](cached, cached + 46, indices), 46.)
    np.testing.assert_equal(noise["p90_difference"](cached, cached, indices), 0.)
    with pytest.raises(ValueError, match="observed masks"):
        noise["p90_difference"](cached, np.zeros(11), indices)


def test_balanced_bootstrap_preserves_partial_edge_blocks_without_last_frame_duplication(noise):
    indices = noise["balanced_block_indices"](11, 4, 20000, np.random.default_rng(1729), 2)
    valid = indices[indices >= 0]
    # Each original frame has expected multiplicity one despite short first/last blocks.
    np.testing.assert_allclose(np.bincount(valid, minlength=11) / 20000, np.ones(11), atol=.025)
    blocks = indices.reshape(20000, -1, 4)
    for block in np.unique(blocks.reshape(-1, 4), axis=0):
        np.testing.assert_equal(np.diff(block[block >= 0]), 1)
    cached: np.ndarray = np.arange(11, dtype=float)
    np.testing.assert_allclose(noise["p90_difference"](cached, cached + 46, indices), 46.)


def test_bootstrap_explicitly_counts_undefined_replicates_instead_of_zero_error(noise):
    indices = np.array([[0, 0], [0, 1]])
    samples = noise["p90_difference"](np.array([np.nan, 1.]), np.array([np.nan, 47.]), indices)
    assert np.isnan(samples[0]) and samples[1] == 46
    result = noise["summarize"](samples)
    assert result["defined_replicates"] == result["undefined_no_observed_replicates"] == 1
    assert result["percentile_95_interval_px"] == [46., 46.]
    assert result["std_px"] is None
