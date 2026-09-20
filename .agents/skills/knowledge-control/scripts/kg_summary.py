#!/usr/bin/env python
"""Detect unreviewed knowledge changes; acknowledge only after editing summary.md."""
from __future__ import annotations

import argparse
import hashlib
import re
from datetime import date

from kg_lib import nodes_dir

MARKER = re.compile(r"<!-- knowledge-review: [0-9a-f]{64} on \d{4}-\d{2}-\d{2} -->\n?")


def fingerprint() -> str:
    base = nodes_dir().parent
    paths = sorted([*nodes_dir().rglob("*.md"), *(base / "Papers").glob("*/paper.md")])
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(base)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def check_summary() -> list[str]:
    path = nodes_dir().parent / "summary.md"
    if not path.is_file():
        return ["summary.md is missing"]
    match = MARKER.search(path.read_text())
    if not match or fingerprint() not in match.group():
        return ["summary.md has unreviewed node/paper changes; update the narrative, then kg_summary.py --mark-reviewed"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mark-reviewed", action="store_true", help="record that the summary was reviewed against the current evidence")
    args = parser.parse_args()
    if args.mark_reviewed:
        path = nodes_dir().parent / "summary.md"
        text = MARKER.sub("", path.read_text())
        marker = f"<!-- knowledge-review: {fingerprint()} on {date.today().isoformat()} -->\n"
        path.write_text(marker + text)
    errors = check_summary()
    for error in errors:
        print(f"ERROR: {error}")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
