"""Errors of estimated field and flight-segment parameters."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def _stats(values: NDArray[np.floating]) -> dict[str, float | int | None]:
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
    }


def field_parameter_report(
    predicted: dict[str, NDArray[np.floating]], truth: dict[str, NDArray[np.floating]]
) -> dict[str, Any]:
    """Per-rally errors for ``wind`` ``(R,3)``, ``k_drag`` and ``k_magnus`` ``(R,)``.

    Wind errors are Euclidean in m/s; coefficient errors are relative.
    """
    _require_keys(predicted, truth, ("wind", "k_drag", "k_magnus"))
    return {
        "wind_error_mps": _stats(
            np.linalg.norm(predicted["wind"] - truth["wind"], axis=-1)
        ),
        "k_drag_relative_error": _stats(
            np.abs(predicted["k_drag"] / truth["k_drag"] - 1)
        ),
        "k_magnus_relative_error": _stats(
            np.abs(predicted["k_magnus"] / truth["k_magnus"] - 1)
        ),
    }


def segment_parameter_report(
    predicted: dict[str, NDArray[np.floating]], truth: dict[str, NDArray[np.floating]]
) -> dict[str, Any]:
    """Per-segment errors for ``position``, ``velocity``, ``spin`` ``(S,3)`` and
    the Magnus coefficient vector ``k_magnus * spin`` (``magnus`` ``(S,3)``).

    ``magnus`` is the only spin quantity free flight determines; the angle error
    of spin is reported where both spins exceed 1 rad/s.
    """
    _require_keys(predicted, truth, ("position", "velocity", "spin", "magnus"))
    angle = []
    for a, b in zip(predicted["spin"], truth["spin"], strict=True):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 1 and nb > 1:
            angle.append(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1, 1))))
    return {
        "position_error_m": _stats(
            np.linalg.norm(predicted["position"] - truth["position"], axis=-1)
        ),
        "velocity_error_mps": _stats(
            np.linalg.norm(predicted["velocity"] - truth["velocity"], axis=-1)
        ),
        "spin_error_radps": _stats(
            np.linalg.norm(predicted["spin"] - truth["spin"], axis=-1)
        ),
        "spin_angle_error_deg": _stats(np.array(angle)),
        "magnus_error": _stats(
            np.linalg.norm(predicted["magnus"] - truth["magnus"], axis=-1)
        ),
    }


def _require_keys(
    predicted: dict[str, NDArray[np.floating]],
    truth: dict[str, NDArray[np.floating]],
    keys: tuple[str, ...],
) -> None:
    for key in keys:
        if key not in predicted or key not in truth:
            raise KeyError(f"Missing parameter {key!r}")
        if predicted[key].shape != truth[key].shape:
            raise ValueError(f"Shape mismatch for {key!r}")
