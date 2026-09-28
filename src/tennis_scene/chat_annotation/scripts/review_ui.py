"""Serve the read-only annotation progress and quality dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from ..web.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", required=True, type=Path, help="outputs/chat_annotation の絶対パス"
    )
    parser.add_argument("--port", type=int, default=8769)
    args = parser.parse_args()
    if not args.root.is_absolute():
        parser.error("--root must be absolute")
    uvicorn.run(create_app(args.root), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
