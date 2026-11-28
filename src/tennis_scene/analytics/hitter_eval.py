"""Render prediction-blind multiview evidence and evaluate the V4 hitter rule."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from src.tennis_scene.analytics.hitter_v4 import V4Benchmark

CAMERA_IDS = ("cam0", "cam1", "cam2", "cam3")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    render = commands.add_parser("render", help="render shot panels for visual review")
    render.add_argument("--output", required=True, type=Path)
    render.add_argument("--clip-id", action="append", required=True)
    args = parser.parse_args(argv)

    benchmark = V4Benchmark.load(args.benchmark, args.manifest_sha256)
    if args.command == "render":
        from src.tennis_scene.analytics.hitter_audit import render_audit

        render_audit(benchmark, args.output, args.clip_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
