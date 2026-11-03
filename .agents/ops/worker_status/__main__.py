"""Print a safe, read-only JSON observation of common runner workers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import StatusError, collect_status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=int)
    parser.add_argument(
        "--repo", help="owner/repo; local registry filter, never gh repo view"
    )
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--dispatcher-state-dir", type=Path)
    parser.add_argument("--include-inactive", action="store_true")
    parser.add_argument(
        "--github", action="store_true", help="explicit GET-only GitHub snapshot"
    )
    parser.add_argument(
        "--stale-after",
        type=float,
        default=300,
        help="seconds without received events (default: 300)",
    )
    parser.add_argument("--tail-bytes", type=int, default=262144)
    parser.add_argument("--tool-limit", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        result = collect_status(**vars(args))
    except StatusError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
