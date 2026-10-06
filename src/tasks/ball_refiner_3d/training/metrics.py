"""Physical-coordinate and event evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def event_neighborhood(events: np.ndarray, radius: int = 5) -> NDArray[np.bool_]:
    selected: NDArray[np.bool_] = np.zeros(events.shape, dtype=bool)
    for index in np.flatnonzero(events):
        selected[max(0, index - radius) : min(len(events), index + radius + 1)] = True
    return selected


def error_metrics(
    prediction: np.ndarray, target: np.ndarray, missing: np.ndarray, events: np.ndarray
) -> dict[str, Any]:
    distance = np.linalg.norm(prediction - target, axis=-1)
    if not np.isfinite(distance).all():
        raise ValueError("Nonfinite evaluation error")
    result: dict[str, Any] = {
        "frame_missing_rate": float(missing.mean()),
        "frames": int(missing.size),
    }
    for name, mask in (
        ("all", np.ones_like(missing)),
        ("missing", missing),
        ("observed", ~missing),
        ("event", events),
    ):
        values = distance[mask]
        result[name] = {
            "count": len(values),
            "rmse": float(np.sqrt(np.mean(values**2))) if len(values) else None,
            "mean": float(values.mean()) if len(values) else None,
            "p95": float(np.quantile(values, 0.95)) if len(values) else None,
        }
    return result
