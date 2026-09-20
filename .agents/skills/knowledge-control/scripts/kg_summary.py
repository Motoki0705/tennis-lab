#!/usr/bin/env python
"""Detect unreviewed knowledge changes; acknowledge only after editing summary.md."""
from __future__ import annotations

import argparse
import hashlib
import re
from datetime import date

from kg_lib import nodes_dir
from kg_schema import has_prose, iso_date

MARKER = re.compile(r"<!-- knowledge-review: [0-9a-f]{64} on \d{4}-\d{2}-\d{2} -->\n?")


def fingerprint(narrative: str | None = None) -> str:
    base = nodes_dir().parent
    paths = sorted([*nodes_dir().rglob("*.md"), *(base / "Papers").glob("*/paper.md")])
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(base)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    if narrative is None:
        narrative = (base / "summary.md").read_text(encoding="utf-8")
    digest.update(b"summary.md\0")
    digest.update(MARKER.sub("", narrative).encode())
    return digest.hexdigest()


def check_summary() -> list[str]:
    path = nodes_dir().parent / "summary.md"
    if not path.is_file():
        return ["summary.md is missing"]
    text = path.read_text(encoding="utf-8")
    matches = list(MARKER.finditer(text))
    if len(matches) != 1 or text.count("<!-- knowledge-review:") != 1:
        return ["summary.md has unreviewed changes: exactly one review marker is required; review then kg_summary.py --mark-reviewed"]
    if not iso_date(matches[0].group().split(" on ")[1].split(" ")[0]):
        return ["summary.md review marker must contain a valid ISO date"]
    if not has_prose(MARKER.sub("", text)):
        return ["summary.md needs a substantive narrative, not just headings/comments"]
    if fingerprint(text) not in matches[0].group():
        return ["summary.md has unreviewed node/paper changes; update the narrative, then kg_summary.py --mark-reviewed"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mark-reviewed", action="store_true", help="record that the summary was reviewed against the current evidence")
    args = parser.parse_args()
    if args.mark_reviewed:
        path = nodes_dir().parent / "summary.md"
        text = MARKER.sub("", path.read_text())
        if not has_prose(text):
            print("ERROR: summary.md needs a substantive narrative before marking reviewed")
            return 1
        marker = f"<!-- knowledge-review: {fingerprint(text)} on {date.today().isoformat()} -->\n"
        path.write_text(marker + text)
    errors = check_summary()
    for error in errors:
        print(f"ERROR: {error}")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
