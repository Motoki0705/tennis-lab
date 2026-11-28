"""Render dense Typst tables from the same cells as the text reference."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .table_spec import Table
from .typst_runner import compile_table


def typst_source(table: Table) -> str:
    def quote(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)

    widths = ", ".join(f"{c.width}pt" for c in table.columns)
    alignment = ", ".join(
        "left" if c.digits is None else "right" for c in table.columns
    )
    headings = ", ".join(
        f"table-heading({quote(c.label)}, {quote(c.unit)})" for c in table.columns
    )
    cells = []
    group = ""
    for row, displayed, strong in table.display_rows():
        if row.group and row.group != group:
            cells.append(
                f"table.cell(colspan: {len(table.columns)}, fill: band, align: left, strong({quote(row.group)}))"
            )
        group = row.group
        for column, value, bold in zip(table.columns, displayed, strong, strict=True):
            cell = quote(value)
            if column.digits is not None:
                cell = f"numeric({cell})"
            cells.append(f"strong({cell})" if bold else cell)
    template = Path(__file__).with_name("table.typ").read_text(encoding="utf-8")
    body = ",\n    ".join(cells)
    return (
        f"{template}\n#dense-table(\n"
        f"  ({widths}), ({alignment}), {quote(table.title)}, {quote(table.note)},\n"
        f"  ({headings}), (\n    {body},\n  ),\n)\n"
    )


def render_table(table: Table, typst: Path, target: Path) -> None:
    with TemporaryDirectory(prefix="wiki-typst-") as directory:
        script = Path(directory) / "table.typ"
        script.write_text(typst_source(table), encoding="utf-8")
        compile_table(typst, script, target)
