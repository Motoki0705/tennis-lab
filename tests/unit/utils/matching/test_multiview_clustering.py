"""Multi-view clustering against exhaustive enumeration of valid partitions."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from src.utils.matching import cluster_multiview, decision_margins
from src.utils.matching.multiview_clustering import MAX_ITEMS


def _partitions(count: int):
    """Every set partition of ``range(count)`` as a label tuple (restricted growth strings)."""
    def grow(prefix: list[int]):
        if len(prefix) == count:
            yield tuple(prefix)
            return
        for label in range(max(prefix, default=-1) + 2):
            yield from grow([*prefix, label])
    yield from grow([])


def _brute_force(scores, views, allowed):
    """Objective of every valid partition, keyed by its label tuple."""
    count = len(views)
    values = {}
    for labels in _partitions(count):
        pairs = [(i, j) for i, j in combinations(range(count), 2) if labels[i] == labels[j]]
        if any(views[i] == views[j] or not allowed[i, j] for i, j in pairs):
            continue
        values[labels] = sum(scores[i, j] for i, j in pairs)
    return values


def _random_problem(rng, count, views_count):
    views = rng.integers(0, views_count, count)
    raw = rng.normal(size=(count, count))
    scores = (raw + raw.T) / 2
    allowed = rng.random((count, count)) > .2
    allowed = allowed & allowed.T
    return scores, views, allowed


def test_two_players_seen_by_three_cameras_are_recovered():
    # Items: cam0 {A, B}, cam1 {A, B}, cam2 {B, A}.
    views = np.array([0, 0, 1, 1, 2, 2])
    person = np.array([0, 1, 0, 1, 1, 0])
    scores = np.where(person[:, None] == person[None, :], 1., -1.)
    result = cluster_multiview(scores, views)
    np.testing.assert_array_equal(result.labels, [0, 1, 0, 1, 1, 0])
    assert result.objective == pytest.approx(6.)
    assert not result.linked.diagonal().any()


def test_same_view_items_never_share_an_identity_even_when_scored_high():
    views = np.array([0, 0, 1])
    scores = np.full((3, 3), 5.)
    result = cluster_multiview(scores, views)
    assert result.labels[0] != result.labels[1]
    assert result.linked[2].sum() == 1
    assert result.objective == pytest.approx(5.)


def test_a_negative_pair_is_linked_when_its_identity_outweighs_it():
    # A-B and B-C strongly similar, A-C mildly dissimilar: correlation clustering keeps all three together.
    scores = np.array([[0., 2., -.5], [2., 0., 2.], [-.5, 2., 0.]])
    result = cluster_multiview(scores, np.array([0, 1, 2]))
    assert len(set(result.labels.tolist())) == 1
    assert result.objective == pytest.approx(3.5)


def test_allowed_mask_forbids_a_link_outright():
    scores = np.array([[0., 3.], [3., 0.]])
    forbidden = np.array([[True, False], [False, True]])
    result = cluster_multiview(scores, np.array([0, 1]), allowed=forbidden, with_margins=True)
    assert result.labels.tolist() == [0, 1]
    assert result.margins is not None and np.isinf(result.margins[0, 1])


@pytest.mark.parametrize("seed", range(12))
def test_objective_and_margins_match_exhaustive_enumeration(seed):
    rng = np.random.default_rng(seed)
    scores, views, allowed = _random_problem(rng, count=int(rng.integers(2, 8)), views_count=3)
    values = _brute_force(scores, views, allowed)
    result = cluster_multiview(scores, views, allowed=allowed, with_margins=True)
    margins = result.margins
    assert margins is not None
    best = max(values.values())
    assert result.objective == pytest.approx(best, abs=1e-9)
    assert values[tuple(result.labels.tolist())] == pytest.approx(best, abs=1e-9)
    count = len(views)
    for i, j in combinations(range(count), 2):
        if views[i] == views[j] or not allowed[i, j]:
            assert np.isinf(margins[i, j])
            continue
        flipped = [v for labels, v in values.items() if (labels[i] == labels[j]) != bool(result.linked[i, j])]
        assert margins[i, j] == pytest.approx(best - max(flipped), abs=1e-9)
        assert margins[i, j] == margins[j, i]
    # The smallest decision margin is the gap to the second-best partition.
    ranked = sorted(values.values(), reverse=True)
    finite = margins[np.isfinite(margins)]
    if len(ranked) > 1 and len(finite):
        assert finite.min() == pytest.approx(ranked[0] - ranked[1], abs=1e-9)


def test_tied_assignment_is_reported_as_ambiguous():
    # cam1 track 2 is equally similar to cam0 tracks 0 and 1: either link is optimal.
    views = np.array([0, 0, 1])
    scores = np.array([[0., -1., 1.], [-1., 0., 1.], [1., 1., 0.]])
    result = cluster_multiview(scores, views, with_margins=True)
    ambiguous = result.ambiguous_pairs(.1)
    assert {(i, j) for i, j, _ in ambiguous} == {(0, 2), (1, 2)}
    assert all(margin == pytest.approx(0.) for _, _, margin in ambiguous)
    # Every tied pair touches item 2, so restricting to it keeps all of them.
    assert result.ambiguous_pairs(.1, items=np.array([False, False, True])) == ambiguous
    assert result.ambiguous_pairs(0.) == []


def test_clear_assignment_has_margins_equal_to_score_gaps():
    views = np.array([0, 0, 1, 1])
    scores = np.zeros((4, 4))
    scores[0, 2] = scores[2, 0] = 3.
    scores[1, 3] = scores[3, 1] = 2.
    scores[0, 3] = scores[3, 0] = -1.
    scores[1, 2] = scores[2, 1] = -1.
    result = cluster_multiview(scores, views, with_margins=True)
    margins = result.margins
    assert margins is not None
    assert result.labels.tolist() == [0, 1, 0, 1]
    assert margins[0, 2] == pytest.approx(3.)
    assert margins[1, 3] == pytest.approx(2.)
    # Linking 0-3 forces 0-2 and 1-3 apart (view exclusivity): 5 - (-1) = 6.
    assert margins[0, 3] == pytest.approx(6.)
    assert result.ambiguous_pairs(2.5) == [(1, 3, pytest.approx(2.))]


def test_degenerate_sizes():
    empty = cluster_multiview(np.zeros((0, 0)), np.zeros(0, np.int64), with_margins=True)
    assert empty.labels.shape == (0,) and empty.objective == 0.
    single = cluster_multiview(np.zeros((1, 1)), np.array([3]), with_margins=True)
    assert single.labels.tolist() == [0]


def test_ambiguity_requires_margins():
    result = cluster_multiview(np.zeros((2, 2)), np.array([0, 1]))
    with pytest.raises(ValueError, match="margins were not computed"):
        result.ambiguous_pairs(.1)


@pytest.mark.parametrize("scores, views, allowed, message", [
    (np.array([[0., 1.], [2., 0.]]), np.array([0, 1]), None, "symmetric"),
    (np.array([[0., np.nan], [np.nan, 0.]]), np.array([0, 1]), None, "finite"),
    (np.zeros((2, 2)), np.array([0., 1.]), None, "integer"),
    (np.zeros((3, 3)), np.array([0, 1]), None, "aligned"),
    (np.zeros((2, 2)), np.array([0, 1]), np.ones((2, 2)), "boolean"),
    (np.zeros((2, 2)), np.array([0, 1]), np.array([[True, True], [False, True]]), "symmetric boolean"),
])
def test_invalid_inputs_are_rejected(scores, views, allowed, message):
    with pytest.raises(ValueError, match=message):
        cluster_multiview(scores, views, allowed=allowed)


def test_problem_size_is_bounded():
    count = MAX_ITEMS + 1
    with pytest.raises(ValueError, match="at most"):
        cluster_multiview(np.zeros((count, count)), np.arange(count))


@pytest.mark.parametrize("seed", range(6))
def test_requested_margins_equal_the_full_computation(seed):
    rng = np.random.default_rng(100 + seed)
    scores, views, allowed = _random_problem(rng, 6, 3)
    full = cluster_multiview(scores, views, allowed=allowed, with_margins=True)
    base = cluster_multiview(scores, views, allowed=allowed)
    items: np.ndarray = np.zeros(6, bool)
    items[rng.integers(0, 6)] = True
    partial = decision_margins(base, scores, views, allowed=allowed, items=items)
    assert full.margins is not None and partial.margins is not None
    relevant = items[:, None] | items[None, :]
    np.testing.assert_allclose(partial.margins[relevant], full.margins[relevant])
    finite_elsewhere = ~relevant & np.isfinite(full.margins)
    assert np.isnan(partial.margins[finite_elsewhere]).all()
    assert partial.ambiguous_pairs(np.inf, items=items) == full.ambiguous_pairs(np.inf, items=items)


def test_ambiguity_of_an_unrequested_pair_is_refused():
    views = np.array([0, 1, 2])
    scores = np.ones((3, 3))
    base = cluster_multiview(scores, views)
    partial = decision_margins(base, scores, views, items=np.array([True, False, False]))
    with pytest.raises(ValueError, match="were not computed"):
        partial.ambiguous_pairs(1.)


def test_margins_of_a_foreign_clustering_are_refused():
    views = np.array([0, 1])
    base = cluster_multiview(np.array([[0., 1.], [1., 0.]]), views)
    with pytest.raises(ValueError, match="does not belong"):
        decision_margins(base, np.array([[0., -1.], [-1., 0.]]), views)
