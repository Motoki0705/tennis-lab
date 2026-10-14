"""Supervisor-owned ball annotation learning stage, invoked by Learning."""

import argparse
from pathlib import Path

from src.tasks.base.annotation.artifacts import annotation_reference
from src.tasks.base.annotation.paths import AnnotationLayout
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathRole,
)

from .stages import execute

PATH_BOUNDARY = NonHydraPathBoundary(
    name="ball_detection.annotation_learning",
    fields=(
        BoundaryPathField(
            "input",
            PathRole.OUTPUT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
        ),
        BoundaryPathField(
            "output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.FILE
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    layout = AnnotationLayout.production()
    paths = PATH_BOUNDARY.validate(
        {"input": args.input, "output": args.output}, resolver=layout.resolver
    )
    request = annotation_reference(layout, paths.declared("input").path)
    if request.digest != args.input_sha256:
        raise ValueError("ball learning stage input hash changed")
    result = execute(layout, request, paths.declared("output").path)
    print(result.canonical_bytes().decode())


if __name__ == "__main__":
    main()
