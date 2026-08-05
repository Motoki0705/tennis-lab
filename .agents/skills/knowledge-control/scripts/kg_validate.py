#!/usr/bin/env python
"""Validate formal nodes under ``knowledge/nodes/``.

This validator covers experimental run/group nodes and reviewed
paper/proposal nodes.  Raw literature-radar candidates are validated by
``literature-radar/scripts/radar_ingest.py validate`` instead.
"""

from __future__ import annotations

from collections import Counter

from kg_lib import NODE_TYPES, load_nodes, nodes_dir, validate


def main() -> int:
    directory = nodes_dir()
    try:
        nodes = load_nodes(directory)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    result = validate(nodes)
    for warning in result.warnings:
        print(f"WARN: {warning}")
    for error in result.errors:
        print(f"ERROR: {error}")

    counts = Counter(node.type for node in nodes)
    summary = ", ".join(f"{counts[node_type]} {node_type}" for node_type in sorted(NODE_TYPES))
    print(
        f"\n{len(nodes)} nodes ({summary}) in {directory} — "
        f"{len(result.errors)} error(s), {len(result.warnings)} warning(s)."
    )
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
