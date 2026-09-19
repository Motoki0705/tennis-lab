"""Freeze a seeded subset preserving every parent motion-source split."""

import argparse
from pathlib import Path

from src.tasks.plcs.data.preparation import prepare_subset
from src.tasks.plcs.data.preparation_paths import preparation_resolver
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathRole,
)

PATH_BOUNDARY = NonHydraPathBoundary(
    name="plcs.prepare_subset",
    fields=(
        BoundaryPathField(
            "source",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
        ),
        BoundaryPathField(
            "destination", PathRole.ARTIFACT, PathDirection.OUTPUT, PathKind.DIRECTORY
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--evaluation", type=int, default=200)
    args = parser.parse_args()
    paths = PATH_BOUNDARY.validate(
        {"source": args.source, "destination": args.destination},
        resolver=preparation_resolver(args.source, args.destination),
    )
    prepare_subset(
        paths.declared("source").path,
        paths.declared("destination").path,
        seed=args.seed,
        train=args.train,
        evaluation=args.evaluation,
    )


if __name__ == "__main__":
    main()
