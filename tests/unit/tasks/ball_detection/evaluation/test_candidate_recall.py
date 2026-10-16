"""Recall denominators, source distance, ranking, ties and padded peaks."""

import numpy as np
import pytest

from src.tasks.ball_detection.evaluation.candidate_recall import candidate_recall_counts


def test_misses_ranking_and_ties_are_disjoint_and_unknown_gt_is_excluded():
    xy = np.array([
        [[12, 16], [100, 100]],   # exactly 20 px: top-1 hit
        [[30, 0], [0, 0]],       # wrong ranks above true
        [[50, 0], [0, 0]],       # padded candidate is not a hit
        [[40, 0], [0, 0]],       # all candidates invalid
        [[30, 0], [0, 0]],       # same-score order, not strictly higher score
        [[0, 0], [0, 0]],        # unobserved target, do not score
    ], np.float32)
    scores = np.array([[.8, .2], [.8, .01], [.8, 0], [0, 0], [.3, .3], [.9, .8]], np.float32)
    valid = np.array([[1, 1], [1, 1], [1, 0], [0, 0], [1, 1], [1, 1]], bool)
    gt: np.ndarray = np.zeros((6, 2), np.float32)
    gt[-1] = np.nan
    result = candidate_recall_counts(xy, scores, valid, gt, np.array([1, 1, 1, 1, 1, 0], bool), radius_px=20)
    assert result.observed == 5
    assert result.recalled_at_1 == 1
    assert result.recalled_at_k == 3
    assert result.wrong_ranked_above_true == 2
    assert result.wrong_strictly_higher_score == 1
    report = result.report()
    assert report['recall_at_k'] == 3 / 5
    assert report['not_in_candidates_rate'] == 2 / 5
    assert report['wrong_ranked_above_true_rate'] == 2 / 5
    assert report['wrong_ranked_above_true_given_recalled'] == 2 / 3
    assert report['rank_only_due_to_tie'] == 1


def test_uses_score_order_ignores_invalid_slots_and_keeps_tiny_scores():
    xy = np.array([[[np.nan, np.nan], [0, 0], [0, 30]]], np.float32)
    scores = np.array([[np.nan, 1e-8, 1e-7]], np.float32)
    result = candidate_recall_counts(xy, scores, np.array([[0, 1, 1]], bool), np.zeros((1, 2), np.float32),
                                    np.ones(1, bool), radius_px=20)
    assert result.recalled_at_k == 1
    assert result.recalled_at_1 == 0
    assert result.wrong_strictly_higher_score == 1


def test_zero_observed_is_undefined_instead_of_perfect_recall():
    result = candidate_recall_counts(np.zeros((2, 8, 2), np.float32), np.zeros((2, 8), np.float32),
                                    np.zeros((2, 8), bool), np.full((2, 2), np.nan, np.float32),
                                    np.zeros(2, bool), radius_px=20)
    assert result.frames == 2
    assert result.observed == 0
    assert result.report()['recall_at_k'] is None
    assert result.report()['wrong_ranked_above_true_given_recalled'] is None


@pytest.mark.parametrize('radius', [0, -1, np.nan, np.inf])
def test_invalid_radius_fails(radius):
    with pytest.raises(ValueError, match='radius'):
        candidate_recall_counts(np.zeros((1, 8, 2), np.float32), np.zeros((1, 8), np.float32),
                                np.ones((1, 8), bool), np.zeros((1, 2), np.float32), np.ones(1, bool), radius_px=radius)


def test_observed_nonfinite_target_fails():
    with pytest.raises(ValueError, match='finite'):
        candidate_recall_counts(np.zeros((1, 8, 2), np.float32), np.zeros((1, 8), np.float32),
                                np.ones((1, 8), bool), np.full((1, 2), np.nan, np.float32), np.ones(1, bool), radius_px=20)
