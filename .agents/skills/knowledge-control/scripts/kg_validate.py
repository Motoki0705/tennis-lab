#!/usr/bin/env python
"""Validate the knowledge graph under ``knowledge/nodes/``.

Checks frontmatter schema (id/type/title/status/...) and that every edge
reference (``parents``, group ``members``, ``relations[].to``) resolves to an
existing node. Exits non-zero if any error is found so it can gate CI / commits.

Usage:
    .venv/bin/python .agents/skills/knowledge-control/scripts/kg_validate.py
"""

from __future__ import annotations

import argparse

from kg_lib import load_nodes, nodes_dir, validate
from kg_papers import validate_papers
from kg_storage import validate_counters
from kg_summary import check_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-summary", action="store_true")
    args = parser.parse_args()
    directory = nodes_dir()
    try:
        nodes = load_nodes(directory)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    res = validate(nodes)
    res.errors.extend(validate_counters(nodes))
    papers = validate_papers(nodes)
    res.errors.extend(papers.errors)
    if args.check_summary:
        res.errors.extend(check_summary())
    for w in res.warnings:
        print(f"WARN: {w}")
    for e in res.errors:
        print(f"ERROR: {e}")

    runs = sum(1 for n in nodes if n.type == "run")
    groups = sum(1 for n in nodes if n.type == "group")
    print(
        f"\n{len(nodes)} nodes ({runs} run, {groups} group) in {directory} — "
        f"{len(res.errors)} error(s), {len(res.warnings)} warning(s)."
    )
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
