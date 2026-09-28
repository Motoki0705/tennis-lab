"""Frame-weighted percentile intervals with whole temporal camera groups resampled."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Statistic = Literal["mean", "median", "p95"]


def _statistic(values: NDArray[np.float64], statistic: Statistic) -> float:
    if statistic == "mean":
        return float(values.mean())
    return float(np.quantile(values, .5 if statistic == "median" else .95))


def grouped_intervals(
    quantities: Mapping[str, tuple[NDArray[np.float64], Statistic]], groups: Sequence[str], *,
    repetitions: int, confidence: float, seed: int,
) -> dict[str, dict[str, float]]:
    """Resample groups with replacement and retain every frame/camera in each draw.

    Ratios use the number of drawn frames, not the number of clips. All reported
    quantities share the same resamples. Input order does not change grouping.
    """
    if repetitions < 2 or not math.isfinite(confidence) or not 0 < confidence < 1 or seed < 0:
        raise ValueError("Invalid bootstrap repetitions, confidence, or seed")
    unique = sorted(set(groups))
    if len(unique) < 2 or any(not group for group in groups) or not quantities:
        raise ValueError("Bootstrap needs quantities and at least two nonempty group IDs")
    for values, statistic in quantities.values():
        if values.shape != (len(groups),) or not np.isfinite(values).all() or statistic not in ("mean", "median", "p95"):
            raise ValueError("Bootstrap quantities must be finite frame vectors with a known statistic")
    labels = np.asarray(groups)
    indices = [np.flatnonzero(labels == group) for group in unique]
    rng = np.random.default_rng(seed)
    draws: dict[str, NDArray[np.float64]] = {key: np.empty(repetitions, dtype=np.float64) for key in quantities}
    for iteration in range(repetitions):
        selected = np.concatenate([indices[index] for index in rng.integers(len(unique), size=len(unique))])
        for name, (values, statistic) in quantities.items():
            draws[name][iteration] = _statistic(values[selected], statistic)
    tail = (1 - confidence) / 2
    return {
        name: {"value": _statistic(values, statistic),
               "ci_low": float(np.quantile(draws[name], tail)),
               "ci_high": float(np.quantile(draws[name], 1 - tail))}
        for name, (values, statistic) in quantities.items()
    }
