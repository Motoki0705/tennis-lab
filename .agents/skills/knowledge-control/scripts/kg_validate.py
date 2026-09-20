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

from kg_history import validate_history
from kg_lib import load_nodes, nodes_dir, validate
from kg_papers import validate_papers
from kg_storage import validate_counters
from kg_summary import check_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-summary", action="store_true")
    parser.add_argument("--base-ref", help="Git base for immutable node identity and monotonic sequence checks")
    args = parser.parse_args()
    directory = nodes_dir()
    try:
        if not directory.is_dir():
            raise ValueError(f"{directory}: missing nodes directory")
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"{path}: symlinks are not allowed in node storage")
        nodes = load_nodes(directory)
        res = validate(nodes)
        res.errors.extend(validate_counters(nodes))
        papers = validate_papers(nodes)
        res.errors.extend(papers.errors)
        if args.check_summary:
            res.errors.extend(check_summary())
        if args.base_ref:
            res.errors.extend(validate_history(nodes, args.base_ref))
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 1
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
