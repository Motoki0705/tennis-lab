"""A shared numeric projection for the table, charts, and copyable text."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal

from .sources import require

DASH = "—"
SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
Value = str | Decimal | None
Direction = Literal["min", "max"]


@dataclass(frozen=True)
class Column:
    label: str
    width: int
    digits: int | None = None
    unit: str = ""
    scale: int = 1
    best: Direction | None = None

    @property
    def heading(self) -> str:
        return f"{self.label} ({self.unit})" if self.unit else self.label

    def display(self, value: Value) -> str:
        if value is None:
            return DASH
        if self.digits is None and isinstance(value, str):
            return value
        if (
            self.digits is None
            or not isinstance(value, Decimal)
            or not value.is_finite()
        ):
            raise ValueError(f"{self.label}: invalid cell type or value")
        rounded = (value * self.scale).quantize(
            Decimal(1).scaleb(-self.digits), rounding=ROUND_HALF_UP
        )
        return f"{rounded:.{self.digits}f}"


@dataclass(frozen=True)
class Row:
    key: str
    cells: tuple[Value, ...]
    group: str = ""
    exclusion: str | None = None


@dataclass(frozen=True)
class Table:
    slug: str
    title: str
    note: str
    columns: tuple[Column, ...]
    rows: tuple[Row, ...]

    def __post_init__(self) -> None:
        require(SLUG.fullmatch(self.slug) is not None, "Invalid table slug")
        require(
            bool(self.title and self.note and self.rows),
            f"{self.slug}: empty title, note, or table",
        )
        require(
            len({r.key for r in self.rows}) == len(self.rows),
            f"{self.slug}: duplicate row keys",
        )
        require(len(self.columns) >= 2, f"{self.slug}: expected at least two columns")
        for row in self.rows:
            require(
                len(row.cells) == len(self.columns),
                f"{self.slug}: wrong number of cells",
            )
            for column, value in zip(self.columns, row.cells, strict=True):
                column.display(value)

    def best_values(self) -> dict[int, Decimal]:
        """Rank raw values of comparable rows and keep exact ties."""
        best = {}
        for i, column in enumerate(self.columns):
            if column.best is None:
                continue
            values = [r.cells[i] for r in self.rows if r.exclusion is None]
            numeric = [value for value in values if isinstance(value, Decimal)]
            if numeric:
                best[i] = (max if column.best == "max" else min)(numeric)
        return best

    def display_rows(self) -> Iterable[tuple[Row, tuple[str, ...], tuple[bool, ...]]]:
        best = self.best_values()
        for row in self.rows:
            cells = tuple(
                c.display(v) for c, v in zip(self.columns, row.cells, strict=True)
            )
            bold = tuple(
                row.exclusion is None and i in best and value == best[i]
                for i, value in enumerate(row.cells)
            )
            yield row, cells, bold


def escape_markdown(text: str) -> str:
    return re.sub(r"([\\`*_[\]<>|&])", r"\\\1", " ".join(text.split()))


def markdown_table(table: Table) -> str:
    headings = [escape_markdown(c.heading) for c in table.columns]
    lines = [
        "| " + " | ".join(headings) + " |",
        "| "
        + " | ".join("---:" if c.digits is not None else "---" for c in table.columns)
        + " |",
    ]
    group = ""
    for row, cells, bold in table.display_rows():
        if row.group and row.group != group:
            lines.append(
                "| **"
                + escape_markdown(row.group)
                + "** |"
                + " |" * (len(headings) - 1)
            )
        group = row.group
        escaped = [escape_markdown(v) for v in cells]
        lines.append(
            "| "
            + " | ".join(
                f"**{v}**" if strong else v
                for v, strong in zip(escaped, bold, strict=True)
            )
            + " |"
        )
    return "\n".join(lines)
