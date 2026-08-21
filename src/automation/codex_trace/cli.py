"""Command-line entry point for local Codex rollout-trace analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.automation.codex_trace.analyzer import TraceAnalyzer
from src.automation.codex_trace.bundle import TraceBundle, TraceBundleError
from src.automation.codex_trace.classifier import SemanticClassifier
from src.automation.codex_trace.output import (
    render_summary,
    write_json_report,
    write_sqlite_report,
)
from src.automation.codex_trace.visualization import (
    write_html_report,
    write_png_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Analyze a local Codex rollout-trace bundle. Exact provider totals "
            "and estimated item/cluster attribution remain separate."
        )
    )
    parser.add_argument(
        "trace_bundle", type=Path, help="bundle directory or state.json"
    )
    parser.add_argument("--json", type=Path, dest="json_path", help="write full JSON")
    parser.add_argument(
        "--sqlite", type=Path, dest="sqlite_path", help="write normalized SQLite"
    )
    parser.add_argument(
        "--html",
        type=Path,
        dest="html_path",
        help="write a self-contained interactive HTML/SVG report",
    )
    parser.add_argument(
        "--png",
        type=Path,
        dest="png_path",
        help="write a static PNG dashboard suitable for PRs",
    )
    parser.add_argument(
        "--cluster-rules",
        type=Path,
        help="JSON taxonomy overriding the deterministic default classifier",
    )
    parser.add_argument(
        "--no-reduce",
        action="store_true",
        help="fail instead of running 'codex debug trace-reduce' when state.json is absent",
    )
    parser.add_argument(
        "--codex-binary",
        default="codex",
        help="Codex executable used for trace reduction (default: codex)",
    )
    parser.add_argument(
        "--force", action="store_true", help="replace existing JSON/SQLite outputs"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run analysis and return a process exit code."""

    args = build_parser().parse_args(argv)
    try:
        classifier = (
            SemanticClassifier.from_json(args.cluster_rules)
            if args.cluster_rules is not None
            else SemanticClassifier()
        )
        bundle = TraceBundle.load(
            args.trace_bundle,
            auto_reduce=not args.no_reduce,
            codex_binary=args.codex_binary,
        )
        report = TraceAnalyzer(bundle, classifier=classifier).analyze()
        if args.json_path is not None:
            write_json_report(report, args.json_path, force=args.force)
        if args.sqlite_path is not None:
            write_sqlite_report(report, args.sqlite_path, force=args.force)
        if args.html_path is not None:
            write_html_report(report, args.html_path, force=args.force)
        if args.png_path is not None:
            write_png_report(report, args.png_path, force=args.force)
    except (FileExistsError, OSError, TraceBundleError, ValueError) as exc:
        print(f"codex-trace-analysis: error: {exc}", file=sys.stderr)
        return 2

    print(render_summary(report))
    if args.json_path is not None:
        print(f"JSON: {args.json_path.expanduser().resolve()}")
    if args.sqlite_path is not None:
        print(f"SQLite: {args.sqlite_path.expanduser().resolve()}")
    if args.html_path is not None:
        print(f"HTML: {args.html_path.expanduser().resolve()}")
    if args.png_path is not None:
        print(f"PNG: {args.png_path.expanduser().resolve()}")
    return 0
