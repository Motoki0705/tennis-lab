"""Run the local ball_detection inference UI."""

from __future__ import annotations

import argparse

from src.tasks.base.visualization.detection.cli import (
    add_path_arguments,
    resolve_paths,
    serve,
)
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathRole,
)

PATH_BOUNDARY = NonHydraPathBoundary(
    name="ball_detection.inference_ui",
    fields=(
        BoundaryPathField(
            "project_root",
            PathRole.PROJECT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "data_root",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "outputs_root",
            PathRole.OUTPUT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "checkpoints_root",
            PathRole.CHECKPOINT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            allow_role_root=True,
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_arguments(parser, task="ball_detection", port=8777)
    args = parser.parse_args()
    values, resolver = resolve_paths(args, task="ball_detection")
    PATH_BOUNDARY.validate(values, resolver=resolver)
    serve("ball_detection", "inference", port=args.port, values=values)


if __name__ == "__main__":
    main()
