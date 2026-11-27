"""Verify figure copies and restore them from their pinned Git objects."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parents[1]
COPY_ROW = re.compile(
    r"^\| `(figures/[^ `]+)` \| `([^ `]+)` \| `([0-9a-f]{40})` \| `([0-9a-f]{64})` \|$",
    re.MULTILINE,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--restore",
        action="store_true",
        help="restore copies from the recorded Git objects",
    )
    args = parser.parse_args()
    manifest = PAPER / "evidence" / "figures.md"
    rows = COPY_ROW.findall(manifest.read_text(encoding="utf-8"))
    if not rows or len({destination for destination, *_ in rows}) != len(rows):
        raise SystemExit("Missing or duplicate figure manifest entries")

    for destination, source, revision, expected in rows:
        source_bytes = subprocess.check_output(
            ["git", "show", f"{revision}:{source}"], cwd=ROOT
        )
        if hashlib.sha256(source_bytes).hexdigest() != expected:
            raise SystemExit(f"Source SHA-256 mismatch: {revision}:{source}")
        output = PAPER / destination
        if args.restore:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(source_bytes)
        if hashlib.sha256(output.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"Copy SHA-256 mismatch: {destination}")
    print(f"{len(rows)} figure copies match their Git sources and SHA-256 values")


if __name__ == "__main__":
    main()
