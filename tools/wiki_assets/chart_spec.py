"""Describe chart geometry and comparison eligibility independently of Matplotlib."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .sources import require
from .table_spec import Row, Table


@dataclass(frozen=True)
class Scatter:
    data: Table
    x: int
    y: int
    memory: int | None = None
    zoom: tuple[tuple[float, float], tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        for row in self.data.rows:
            present = all(isinstance(row.cells[i], Decimal) for i in self.metrics)
            require(
                row.exclusion is not None or present,
                f"{self.data.slug}: comparable point is missing a metric",
            )

    @property
    def metrics(self) -> tuple[int, ...]:
        return (
            (self.x, self.y) if self.memory is None else (self.x, self.y, self.memory)
        )

    @property
    def plotted_rows(self) -> tuple[Row, ...]:
        return tuple(
            r
            for r in self.data.rows
            if isinstance(r.cells[self.x], Decimal)
            and isinstance(r.cells[self.y], Decimal)
        )


@dataclass(frozen=True)
class Bars:
    data: Table
    metrics: tuple[int, ...]
    series: int | None = None


Chart = Scatter | Bars


def raw_number(row: Row, column: int) -> Decimal:
    value = row.cells[column]
    if not isinstance(value, Decimal):
        raise ValueError(f"{row.key}: expected a plotted number")
    return value


def pareto_frontier(spec: Scatter) -> tuple[Row, ...]:
    """Minimize latency and optional memory, and maximize accuracy, within one cohort."""
    objectives = {
        row.key: (raw_number(row, spec.x), -raw_number(row, spec.y))
        + (() if spec.memory is None else (raw_number(row, spec.memory),))
        for row in spec.plotted_rows
        if row.exclusion is None
    }
    frontier = []
    for row in spec.plotted_rows:
        if row.key not in objectives:
            continue
        target = objectives[row.key]
        if not any(
            other != target and all(a <= b for a, b in zip(other, target, strict=True))
            for other in objectives.values()
        ):
            frontier.append(row)
    return tuple(
        sorted(
            frontier,
            key=lambda row: (raw_number(row, spec.x), raw_number(row, spec.y), row.key),
        )
    )
