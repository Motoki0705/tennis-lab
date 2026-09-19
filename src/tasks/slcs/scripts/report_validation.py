"""Render CPU-only PR figures from completed, matched SLCS validation bundles."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tasks.slcs.evaluation.pr_report import generate_report
from src.tasks.slcs.scripts._paths import cli_resolver
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathRole,
)

PATH_BOUNDARY = NonHydraPathBoundary(
    name="slcs.report_validation",
    fields=(
        BoundaryPathField(
            "evaluations",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.ANY,
            allow_role_root=True,
            many=True,
        ),
        BoundaryPathField(
            "output_root",
            PathRole.OUTPUT,
            PathDirection.OUTPUT,
            PathKind.ANY,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "output",
            PathRole.OUTPUT,
            PathDirection.OUTPUT,
            PathKind.ANY,
            allow_role_root=True,
        ),
    ),
)


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
    evaluations = _named_paths(args.evaluation)
    training = None if args.training is None else _named_paths(args.training)
    inputs = tuple(evaluations.values()) + (
        () if training is None else tuple(training.values())
    )
    resolver = cli_resolver(args.output_root)
    PATH_BOUNDARY.validate(
        {
            "evaluations": tuple(dict.fromkeys(inputs)),
            "output_root": args.output_root,
            "output": resolver.resolve(PathRole.OUTPUT, args.output),
        },
        resolver=resolver,
        independent_artifact_inputs=True,
    )
    print(
        generate_report(
            evaluations=evaluations,
            training=training,
            output_root=args.output_root,
            output=args.output,
        )
    )


if __name__ == "__main__":
    main()
