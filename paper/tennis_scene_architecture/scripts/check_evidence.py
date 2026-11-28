"""Check that the LaTeX evidence IDs, source pins, and figure references resolve."""

from __future__ import annotations

import re
from collections import Counter

from make_tables import PAPER, source_bytes, sources

EVIDENCE = re.compile(r"\\ev\{([a-z0-9-]+)\}\{([^{}]+)\}")
LABEL = re.compile(r"\\label\{((?:fig|tab):[^}]+)\}")
REFERENCE = re.compile(r"\\ref\{((?:fig|tab):[^}]+)\}")


def read_rows(section: str, id_pattern: str) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for line in section.splitlines():
        if not line.startswith("| `"):
            continue
        keys = re.findall(id_pattern, line.split("|")[1])
        source_ids = re.findall(r"`([a-z0-9-]+)`", line.split("|")[-2])
        if not keys or not source_ids or any(key in rows for key in keys):
            raise ValueError(f"Invalid or duplicate evidence row: {line}")
        for key in keys:
            rows[key] = source_ids
    return rows


def main() -> None:
    document = (PAPER / "evidence" / "claims.md").read_text(encoding="utf-8")
    _, body = document.split("## 本文と表\n", 1)
    body, figures = body.split("## 図の根拠\n", 1)
    figures, _ = figures.split("## 調査・委任時の記述との差分\n", 1)
    claim_rows = read_rows(body, r"`([a-z0-9-]+)`")
    figure_rows = read_rows(figures, r"`(fig:[^`]+)`")
    cited_sources = {
        source for ids in (claim_rows | figure_rows).values() for source in ids
    }
    if cited_sources != sources().keys():
        raise ValueError(
            f"Source inventory differs from the evidence table: {cited_sources ^ sources().keys()}"
        )
    for source in sorted(cited_sources):
        source_bytes(source)

    paths = [PAPER / "main.tex"]
    for directory in ("chapters", "appendix", "tables"):
        paths.extend(sorted((PAPER / directory).glob("*.tex")))
    manuscript = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    uses = Counter(claim_id for claim_id, _ in EVIDENCE.findall(manuscript))
    if uses.keys() != claim_rows.keys():
        raise ValueError(
            f"LaTeX and evidence IDs differ: {uses.keys() ^ claim_rows.keys()}"
        )
    labels = Counter(LABEL.findall(manuscript))
    if set(labels) != set(REFERENCE.findall(manuscript)) or any(
        count != 1 for count in labels.values()
    ):
        raise ValueError(
            "Every table and figure must have a unique label and a reference"
        )
    if set(figure_rows) != {label for label in labels if label.startswith("fig:")}:
        raise ValueError("Figure labels and the figure evidence table differ")
    print(
        f"{len(claim_rows)} claim rows, {sum(uses.values())} LaTeX evidence links, "
        f"{len(figure_rows)} figures, {len(labels) - len(figure_rows)} tables, {len(cited_sources)} pinned sources"
    )


if __name__ == "__main__":
    main()
