"""Globally allocate exclusive, time-ordered paths through fixed player cores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix

_START, _END = -1, -2


@dataclass(frozen=True)
class FragmentLink:
    before: int
    after: int
    cost: float
    position_residual_m: float
    velocity_difference_mps: float | None
    log_bone_difference: float
    shared_bones: int


@dataclass(frozen=True)
class SlotAssignment:
    core: int
    fragments: tuple[int, ...]
    links: tuple[tuple[FragmentLink, float], ...]


def reachable_fragments(core: int, links: list[FragmentLink]) -> set[int]:
    """Find fragments on a path through the core, with links sorted by source time."""
    earlier = {core}
    later = {core}
    for link in reversed(links):
        if link.after in earlier:
            earlier.add(link.before)
    for link in links:
        if link.before in later:
            later.add(link.after)
    return earlier | later


def assign_slots(
    cores: tuple[int, ...], rewards: dict[int, float], links: list[FragmentLink]
) -> tuple[SlotAssignment, ...]:
    """Choose one path through each core and use each fragment at most once."""
    # Constraints on flow balance, mandatory cores and exclusive fragment use
    # make every integral solution a set of paths in the time-ordered graph.
    variables: list[tuple[int, int, int, FragmentLink | None]] = []
    objectives = []
    exclusive = {fragment: i for i, fragment in enumerate(rewards)}
    lower: list[float] = [0.0] * len(exclusive)
    upper: list[float] = [1.0] * len(exclusive)
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []

    def add_row(lo: float, hi: float) -> int:
        index = len(lower)
        lower.append(lo)
        upper.append(hi)
        return index

    for slot, core in enumerate(cores):
        other_cores = set(cores) - {core}
        options = [
            link
            for link in links
            if link.before not in other_cores and link.after not in other_cores
        ]
        usable = reachable_fragments(core, options)
        balance = {fragment: add_row(0.0, 0.0) for fragment in sorted(usable)}
        start = add_row(1.0, 1.0)
        visit_core = add_row(1.0, 1.0)
        edges: list[tuple[int, int, FragmentLink | None]] = [
            (_START, fragment, None) for fragment in balance
        ]
        edges.extend(
            (link.before, link.after, link)
            for link in options
            if link.before in usable and link.after in usable
        )
        edges.extend((fragment, _END, None) for fragment in balance)
        for before, after, link in edges:
            column = len(variables)
            variables.append((slot, before, after, link))
            # The core contributes a constant. Omit it from the objective so
            # differences in margin reflect the optional fragment evidence.
            reward = rewards[after] if after >= 0 and after != core else 0.0
            objectives.append((link.cost if link is not None else 0.0) - reward)
            constraints = (
                [(start, 1.0)] if before == _START else [(balance[before], -1.0)]
            )
            if after >= 0:
                constraints.extend(((balance[after], 1.0), (exclusive[after], 1.0)))
            if after == core:
                constraints.append((visit_core, 1.0))
            for row, value in constraints:
                rows.append(row)
                cols.append(column)
                values.append(value)

    if not any(link is not None for _, _, _, link in variables):
        return tuple(SlotAssignment(core, (core,), ()) for core in cores)

    matrix = csc_matrix((values, (rows, cols)), shape=(len(lower), len(variables)))
    constraints = LinearConstraint(matrix, lower, upper)
    objective = np.asarray(objectives, dtype=np.float64)

    def solve(exclude: int | None = None) -> NDArray[np.int64]:
        bound: NDArray[np.float64] = np.ones(len(variables), np.float64)
        if exclude is not None:
            bound[exclude] = 0.0
        fit = milp(
            objective,
            integrality=np.ones(len(variables), np.uint8),
            bounds=Bounds(0.0, bound),
            constraints=constraints,
            options={"mip_rel_gap": 0.0},
        )
        if (
            not fit.success
            or fit.x is None
            or not np.allclose(fit.x, fit.x > 0.5, atol=1e-6, rtol=0)
        ):
            raise RuntimeError(
                f"Player slot assignment did not find an integer optimum: {fit.message}"
            )
        return np.flatnonzero(fit.x > 0.5)

    optimum = solve()
    optimum_cost = float(objective[optimum].sum())
    chosen: tuple[dict[int, tuple[int, FragmentLink | None, float]], ...] = tuple(
        {} for _ in cores
    )
    for index in optimum:
        slot, before, after, link = variables[index]
        margin = 0.0
        if link is not None:
            margin = float(objective[solve(int(index))].sum()) - optimum_cost
            if margin < -1e-7:
                raise RuntimeError(
                    "Removing an integration link improved the slot optimum"
                )
        chosen[slot][before] = (after, link, max(0.0, margin))
    result = []
    for core, successors in zip(cores, chosen, strict=True):
        fragment, _, _ = successors[_START]
        path = [fragment]
        used = []
        while True:
            following, link, margin = successors[fragment]
            if link is None:
                break
            used.append((link, margin))
            path.append(following)
            fragment = following
        result.append(SlotAssignment(core, tuple(path), tuple(used)))
    return tuple(result)
