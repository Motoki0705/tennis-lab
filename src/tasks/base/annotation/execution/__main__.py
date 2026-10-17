"""Execute a frozen, resource-accounted stage command."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathRole,
)

from ..paths import AnnotationLayout
from .supervisor import supervise

PATH_BOUNDARY = NonHydraPathBoundary(
    name="base.annotation.execution.cli",
    fields=(
        BoundaryPathField(
            "launch",
            PathRole.OUTPUT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--queued", action="store_true")
    parser.add_argument("--lock-fd", type=int)
    parser.add_argument("--resource-fd", type=int)
    parser.add_argument("--test-root", type=Path)
    args = parser.parse_args()
    layout = (
        AnnotationLayout.production()
        if args.test_root is None
        else AnnotationLayout.testing(args.test_root)
    )
    paths = PATH_BOUNDARY.validate({"launch": args.launch}, resolver=layout.resolver)
    raise SystemExit(
        supervise(
            layout,
            paths.declared("launch").path,
            args.sha256,
            queued=args.queued,
            lock_fd=args.lock_fd,
            resource_fd=args.resource_fd,
        )
    )


if __name__ == "__main__":
    main()
