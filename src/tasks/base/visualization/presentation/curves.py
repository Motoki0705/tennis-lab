"""Progressive learning curves without revealing future measurements."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from src.tasks.base.visualization.presentation import style

if TYPE_CHECKING:
    from matplotlib.axes import Axes


@dataclass(frozen=True)
class CurveSeries:
    """One sorted series; missing/nonfinite values must be handled by its owner."""

    name: str
    steps: NDArray[np.int64]
    values: NDArray[np.float64]
    style: Literal["line", "points"]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("series name must be nonempty")
        if not isinstance(self.steps, np.ndarray) or not isinstance(
            self.values, np.ndarray
        ):
            raise TypeError("steps and values must be numpy arrays")
        if self.steps.dtype != np.int64 or self.values.dtype != np.float64:
            raise TypeError("steps and values must have dtypes int64 and float64")
        if self.steps.ndim != 1 or self.values.ndim != 1:
            raise ValueError("steps and values must be one-dimensional")
        if self.steps.shape != self.values.shape:
            raise ValueError("steps and values must have the same shape")
        if not np.isfinite(self.values).all():
            raise ValueError("curve values must be finite")
        if np.any(self.steps < 0) or np.any(self.steps[1:] <= self.steps[:-1]):
            raise ValueError("steps must be nonnegative and strictly increasing")
        if self.style not in ("line", "points"):
            raise ValueError(f"unknown curve style: {self.style!r}")


def draw_progressive_curves(
    ax: Axes,
    series: Sequence[CurveSeries],
    cursor_step: int,
    *,
    spans: Sequence[tuple[int, int, str]],
    markers: Sequence[tuple[int, float, str]] = (),
) -> None:
    """Append data at or before the cursor, stage spans, markers and a cursor.

    The caller clears the axes between frames, sets limits/axis labels and may
    set a color cycle (for example to match task-specific heads). A series uses
    one cycle color even before its first sample, so its color stays stable.
    ``line`` connects measured samples and ``points`` leaves them unconnected.
    Span labels and marker labels are artist metadata, not on-screen text.
    """
    if isinstance(cursor_step, bool) or not isinstance(cursor_step, int):
        raise TypeError("cursor_step must be an integer")
    if cursor_step < 0:
        raise ValueError("cursor_step must be nonnegative")
    for start, stop, _ in spans:
        if (
            isinstance(start, bool)
            or isinstance(stop, bool)
            or not isinstance(start, int)
            or not isinstance(stop, int)
            or not 0 <= start < stop
        ):
            raise ValueError("span bounds must be integers with 0 <= start < stop")
    for step, value, _ in markers:
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step < 0
            or not math.isfinite(value)
        ):
            raise ValueError(
                "markers require nonnegative integer steps and finite values"
            )

    ax.set_facecolor(style.BACKGROUND)
    ax.tick_params(colors=style.FOREGROUND)
    ax.xaxis.label.set_color(style.FOREGROUND)
    ax.yaxis.label.set_color(style.FOREGROUND)
    for spine in ax.spines.values():
        spine.set_color(style.FIXED)
    for index, (start, stop, label) in enumerate(spans):
        ax.axvspan(
            start,
            stop,
            color=style.FIXED,
            alpha=0.12 if index % 2 == 0 else 0.22,
            label=label,
            zorder=0,
        )
    for curve in series:
        count = int(np.searchsorted(curve.steps, cursor_step, side="right"))
        ax.plot(
            curve.steps[:count],
            curve.values[:count],
            linestyle="-" if curve.style == "line" else "None",
            marker="o" if curve.style == "points" else None,
            markersize=4,
            linewidth=1.8,
            label=curve.name,
            zorder=2,
        )
    for step, value, label in markers:
        if step <= cursor_step:
            ax.axvline(
                step, color=style.FOREGROUND, linestyle="--", alpha=0.5, zorder=1
            )
            ax.plot(
                step,
                value,
                marker="*",
                color=style.NEW,
                markersize=11,
                linestyle="None",
                label=label,
                zorder=4,
            )
    ax.axvline(cursor_step, color=style.FOREGROUND, linewidth=1, alpha=0.8, zorder=3)
