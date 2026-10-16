"""Exact identity clustering of items seen from several views, on any pairwise score.

Items are, for example, camera-local person tracks. A clustering links items
into identities so that

* no identity holds two items of the same view (view exclusivity), and
* linking is transitive (a partition: ``i~j`` and ``j~k`` imply ``i~k``).

Among such partitions the solver returns one that maximizes the sum of
``scores[i, j]`` over linked pairs (correlation clustering), solved exactly as
a MILP. A positive score favours linking, a negative one opposes it; a pair
can still be linked against a negative score when the rest of its identity
outweighs it. Only ``allowed`` forbids a pair outright.

Ambiguity is measured per pair: the *decision margin* of ``(i, j)`` is how
much the objective drops when that one link decision is forced the other way
and everything else is re-optimized. The smallest margin equals the gap
between the best and the second-best partition, so callers can refuse to act
on a clustering whose relevant decisions are nearly tied.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

MAX_ITEMS = 40
"""The transitivity constraints grow as ``N^3``; beyond this the problem is not a view-association problem."""


class SolverTimeLimit(RuntimeError):
    """The MILP did not prove optimality within its time limit; no clustering is returned."""


@dataclass(frozen=True)
class MultiviewClustering:
    """An optimal partition and, optionally, the margin of every pair decision.

    ``labels`` numbers identities densely in order of their first item.
    ``linked[i, j]`` is the pair decision. ``margins[i, j]`` is the objective
    drop of flipping it (``inf`` where the flip is impossible: the diagonal,
    same-view and forbidden pairs; ``nan`` where it was not computed because
    neither item was requested); ``None`` unless margins were requested.
    """

    labels: NDArray[np.int64]  # (N,)
    linked: NDArray[np.bool_]  # (N, N), symmetric
    objective: float
    margins: NDArray[np.float64] | None = None  # (N, N), symmetric

    def ambiguous_pairs(self, min_margin: float, items: NDArray[np.bool_] | None = None) -> list[tuple[int, int, float]]:
        """Pairs whose decision margin is below ``min_margin``, restricted to ``items`` if given.

        A pair is relevant when either of its items is selected. Sorted by margin.
        """
        if self.margins is None:
            raise ValueError("Decision margins were not computed; call cluster_multiview(..., with_margins=True)")
        if not min_margin >= 0:
            raise ValueError("min_margin must be nonnegative")
        count = len(self.labels)
        selected = np.ones(count, bool) if items is None else np.asarray(items, bool)
        if selected.shape != (count,):
            raise ValueError("items must be a boolean (N,) mask")
        relevant = [(i, j) for i, j in combinations(range(count), 2) if selected[i] or selected[j]]
        missing = [(i, j) for i, j in relevant if np.isnan(self.margins[i, j])]
        if missing:
            raise ValueError(f"Decision margins of pairs {missing[:5]} were not computed; request their items in decision_margins")
        pairs = [(i, j, float(self.margins[i, j])) for i, j in relevant if self.margins[i, j] < min_margin]
        return sorted(pairs, key=lambda pair: pair[2])


@lru_cache(maxsize=64)
def _transitivity(count: int) -> tuple[tuple[tuple[int, int], ...], LinearConstraint | None]:
    """Pair variables of ``count`` items and the triangle inequalities ``x_ij + x_ik - x_jk <= 1``."""
    pairs = tuple(combinations(range(count), 2))
    if count < 3:
        return pairs, None
    index = {pair: e for e, pair in enumerate(pairs)}
    rows: list[int] = []
    columns: list[int] = []
    coefficients: list[float] = []
    row = 0
    for i, j, k in combinations(range(count), 3):
        edges = (index[i, j], index[i, k], index[j, k])
        for negative in range(3):
            for position, edge in enumerate(edges):
                rows.append(row)
                columns.append(edge)
                coefficients.append(-1. if position == negative else 1.)
            row += 1
    matrix = coo_matrix((coefficients, (rows, columns)), shape=(row, len(pairs))).tocsc()
    return pairs, LinearConstraint(matrix, np.full(row, -np.inf), np.ones(row))


def _validate(scores: NDArray[np.float64], views: NDArray[np.int64], allowed: NDArray[np.bool_] | None
              ) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.bool_]]:
    scores = np.asarray(scores, np.float64)
    views = np.asarray(views)
    count = len(views)
    if views.ndim != 1 or not np.issubdtype(views.dtype, np.integer):
        raise ValueError("views must be an integer (N,) array")
    if count > MAX_ITEMS:
        raise ValueError(f"Multi-view clustering supports at most {MAX_ITEMS} items, got {count}")
    if scores.shape != (count, count):
        raise ValueError("scores must be (N, N) aligned to views")
    off_diagonal = ~np.eye(count, dtype=bool)
    if not np.isfinite(scores[off_diagonal]).all() or not np.array_equal(scores, scores.T, equal_nan=True):
        raise ValueError("scores must be finite and symmetric off the diagonal")
    permitted: NDArray[np.bool_]
    if allowed is None:
        permitted = np.ones((count, count), bool)
    else:
        permitted = np.asarray(allowed)
        if permitted.shape != (count, count) or permitted.dtype != np.bool_ or not np.array_equal(permitted, permitted.T):
            raise ValueError("allowed must be a symmetric boolean (N, N) mask")
    permitted = permitted & (views[:, None] != views[None, :]) & off_diagonal
    return scores, views.astype(np.int64), permitted


def _solve(scores: NDArray[np.float64], permitted: NDArray[np.bool_], forced: tuple[int, int, bool] | None,
           time_limit_s: float) -> tuple[NDArray[np.bool_], float]:
    """Optimal pair decisions (as an (N, N) mask) and objective, optionally with one decision forced."""
    count = len(scores)
    pairs, constraints = _transitivity(count)
    linked: NDArray[np.bool_] = np.zeros((count, count), bool)
    if not pairs:
        return linked, 0.
    weights = np.asarray([scores[i, j] for i, j in pairs])
    lower = np.zeros(len(pairs))
    upper = np.asarray([float(permitted[i, j]) for i, j in pairs])
    if forced is not None:
        edge = pairs.index((forced[0], forced[1]))
        lower[edge] = upper[edge] = float(forced[2])
    solution = milp(-weights, integrality=np.ones(len(pairs)), bounds=Bounds(lower, upper),
                    constraints=() if constraints is None else constraints, options={"time_limit": time_limit_s})
    if solution.status == 1:
        raise SolverTimeLimit(f"Multi-view clustering MILP hit its {time_limit_s} s time limit before proving optimality")
    if solution.status != 0 or solution.x is None:
        raise RuntimeError(f"Multi-view clustering MILP did not reach a proven optimum: {solution.message}")
    chosen = solution.x > .5
    for (i, j), link in zip(pairs, chosen, strict=True):
        linked[i, j] = linked[j, i] = link
    return linked, float(weights[chosen].sum())


def _labels(linked: NDArray[np.bool_]) -> NDArray[np.int64]:
    count = len(linked)
    labels: NDArray[np.int64] = np.full(count, -1, np.int64)
    for item in range(count):
        if labels[item] < 0:
            labels[(linked[item] | (np.arange(count) == item))] = labels.max() + 1
    return labels


def cluster_multiview(scores: NDArray[np.float64], views: NDArray[np.int64], *, allowed: NDArray[np.bool_] | None = None,
                      with_margins: bool = False, time_limit_s: float = 10.) -> MultiviewClustering:
    """Maximum-score partition of items into identities with at most one item per view.

    ``scores`` is a symmetric (N, N) matrix (the diagonal is ignored), ``views``
    the view index of every item and ``allowed`` an optional symmetric mask of
    pairs that may be linked. Same-view pairs are never linked. Raises
    :class:`SolverTimeLimit` when the MILP does not prove optimality within
    ``time_limit_s``; no suboptimal clustering is returned.
    """
    scores, views, permitted = _validate(scores, views, allowed)
    linked, objective = _solve(scores, permitted, None, time_limit_s)
    labels = _labels(linked)
    count = len(labels)
    if not np.array_equal((labels[:, None] == labels[None, :]) & ~np.eye(count, dtype=bool), linked):
        raise RuntimeError("MILP link decisions are not transitive")
    clustering = MultiviewClustering(labels, linked, objective)
    return decision_margins(clustering, scores, views, allowed=allowed, time_limit_s=time_limit_s) if with_margins else clustering


def decision_margins(clustering: MultiviewClustering, scores: NDArray[np.float64], views: NDArray[np.int64], *,
                     allowed: NDArray[np.bool_] | None = None, items: NDArray[np.bool_] | None = None,
                     time_limit_s: float = 10.) -> MultiviewClustering:
    """``clustering`` with the decision margins of the pairs touching ``items`` (all pairs if ``None``).

    ``scores``, ``views`` and ``allowed`` must be the problem ``clustering``
    solved. Each margin re-solves the MILP with one decision forced, so a
    caller that only acts on some items (for example the selected players)
    requests just those; the other margins are ``nan`` and
    :meth:`MultiviewClustering.ambiguous_pairs` refuses to judge them.
    """
    scores, views, permitted = _validate(scores, views, allowed)
    count = len(views)
    if clustering.linked.shape != (count, count):
        raise ValueError("The clustering was solved on a different number of items")
    selected = np.ones(count, bool) if items is None else np.asarray(items)
    if selected.shape != (count,) or selected.dtype != np.bool_:
        raise ValueError("items must be a boolean (N,) mask")
    weights = [scores[i, j] for i, j in combinations(range(count), 2) if permitted[i, j]]
    objective = float(sum(scores[i, j] for i, j in combinations(range(count), 2) if clustering.linked[i, j]))
    tolerance = 1e-9 * max(1., float(np.abs(weights).sum()) if weights else 1.)
    if (clustering.linked & ~permitted).any() or abs(objective - clustering.objective) > tolerance:
        raise ValueError("The clustering does not belong to these scores and constraints")
    margins = np.full((count, count), np.inf)
    for i, j in combinations(range(count), 2):
        if not permitted[i, j]:
            continue
        if not (selected[i] or selected[j]):
            margins[i, j] = margins[j, i] = np.nan
            continue
        _, flipped = _solve(scores, permitted, (i, j, not bool(clustering.linked[i, j])), time_limit_s)
        drop = clustering.objective - flipped
        if drop < -tolerance:
            raise RuntimeError(f"Forcing pair ({i}, {j}) improved the objective by {-drop}; the base solution was not optimal")
        margins[i, j] = margins[j, i] = max(drop, 0.)
    return replace(clustering, margins=margins)
