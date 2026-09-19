"""Render CPU-only PR figures from completed, matched SLCS validation bundles."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tasks.slcs.evaluation.pr_report import generate_report


def _named_paths(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        label, separator, path = value.partition("=")
        if (
            not separator
            or not label.strip()
            or not Path(path).is_absolute()
            or label in result
        ):
            raise ValueError("Expected unique LABEL=/absolute/path arguments")
        result[label] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation",
        action="append",
        required=True,
        help="LABEL=/absolute/evaluation/run (one or two)",
    )
    parser.add_argument(
        "--training",
        action="append",
        help="LABEL=/absolute/training/run; supply every label to add learning curves",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--output",
        required=True,
        help="slcs/visualize/<experiment>/<run-id>; must not exist",
    )
    args = parser.parse_args()
    print(
        generate_report(
            evaluations=_named_paths(args.evaluation),
            training=None if args.training is None else _named_paths(args.training),
            output_root=args.output_root,
            output=args.output,
        )
    )


if __name__ == "__main__":
    main()
