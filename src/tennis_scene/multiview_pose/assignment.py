"""Exact weighted set packing over independent groups of conflicting rows."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix


@dataclass(frozen=True)
class Assignment:
    selected: NDArray[np.int64]
    competing: NDArray[np.bool_]
    relative_gap: NDArray[np.float64]
    ambiguous: NDArray[np.bool_]


def _conflict_components(groups: NDArray[np.int64]) -> list[list[int]]:
    parents = list(range(len(groups)))

    def root(index: int) -> int:
        while index != parents[index]:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owners: dict[tuple[int, int], int] = {}
    for index, rows in enumerate(groups):
        for camera in np.flatnonzero(rows >= 0):
            key = int(camera), int(rows[camera])
            previous = owners.setdefault(key, index)
            parents[root(index)] = root(previous)
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(groups)):
        components[root(index)].append(index)
    return list(components.values())


def _optimize(
    scores: NDArray[np.float64], constraints: list[LinearConstraint]
) -> NDArray[np.bool_]:
    fit = milp(
        -scores,
        integrality=np.ones(len(scores), np.int32),
        bounds=Bounds(0.0, 1.0),
        constraints=constraints,
        options={"mip_rel_gap": 0.0},
    )
    if (
        not fit.success
        or fit.x is None
        or not np.isfinite(fit.x).all()
        or not np.allclose(fit.x, (fit.x > 0.5).astype(float), atol=1e-6, rtol=0)
    ):
        raise RuntimeError(
            f"Frame pose assignment did not yield an integer optimum: {fit.message}"
        )
    return np.asarray(fit.x > 0.5, bool)


def _solve_component(
    groups: NDArray[np.int64], scores: NDArray[np.float64]
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], float]:
    indices: dict[tuple[int, int], int] = {}
    rows = []
    columns = []
    for column, group in enumerate(groups):
        for camera in np.flatnonzero(group >= 0):
            key = int(camera), int(group[camera])
            rows.append(indices.setdefault(key, len(indices)))
            columns.append(column)
    usage = csc_matrix(
        (np.ones(len(rows)), (rows, columns)), shape=(len(indices), len(scores))
    )
    exclusive = LinearConstraint(usage, 0.0, 1.0)
    selected = _optimize(scores, [exclusive])
    # Require an unselected hypothesis, so deleting a positive-scoring group
    # from the winner cannot count as an alternative assignment.
    alternative = _optimize(
        scores,
        [exclusive, LinearConstraint((~selected)[None, :].astype(float), 1.0, np.inf)],
    )
    best_score = float(scores[selected].sum())
    runner_up = float(scores[alternative].sum())
    if best_score <= 0 or runner_up > best_score + 1e-7:
        raise RuntimeError("Frame pose assignment objectives disagree")
    return selected, alternative, max(0.0, (best_score - runner_up) / best_score)


def assign(
    groups: NDArray[np.int64],
    scores: NDArray[np.float64],
    ambiguity_relative_gap: float,
) -> Assignment:
    if (
        scores.shape != (len(groups),)
        or not np.isfinite(scores).all()
        or (scores <= 0).any()
    ):
        raise ValueError("Pose hypotheses require finite positive scores")
    selected: NDArray[np.bool_] = np.zeros(len(groups), bool)
    competing = np.zeros_like(selected)
    gaps: NDArray[np.float64] = np.zeros(len(groups))
    for component in _conflict_components(groups):
        if len(component) == 1:
            # Its positive score makes selecting this isolated variable optimal.
            selected[component[0]] = True
            continue
        winner, alternative, gap = _solve_component(
            groups[component], scores[component]
        )
        selected[component] = winner
        replaced = winner & ~alternative
        competing[component] = replaced
        gaps[component] = np.where(replaced, gap, 0.0)
    indices = np.flatnonzero(selected)
    return Assignment(
        indices,
        competing[indices],
        gaps[indices],
        competing[indices] & (gaps[indices] <= ambiguity_relative_gap),
    )
