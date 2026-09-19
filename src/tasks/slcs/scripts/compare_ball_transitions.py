"""Compare two saved SLCS condition directories using an explicit train-only speed threshold."""

import argparse
from pathlib import Path

from src.tasks.slcs.evaluation.ball_transition_comparison import (
    save_ball_transition_comparison,
)
from src.tasks.slcs.scripts._paths import cli_resolver
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathRole,
)

PATH_BOUNDARY = NonHydraPathBoundary(
    name="slcs.compare_ball_transitions",
    fields=(
        BoundaryPathField(
            "baseline",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.ANY,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "candidate",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--fast-speed-mps", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resolver = cli_resolver(args.output.parent)
    PATH_BOUNDARY.validate(
        {"baseline": args.baseline, "candidate": args.candidate, "output": args.output},
        resolver=resolver,
        independent_artifact_inputs=True,
    )
    print(save_ball_transition_comparison(**vars(args)))


if __name__ == "__main__":
    main()
