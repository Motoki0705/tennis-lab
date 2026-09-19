"""Compose existing, time-aligned SLCS renders without running inference."""

import argparse
import sys
from pathlib import Path

from src.tasks.slcs.scripts._paths import cli_resolver
from src.tasks.slcs.visualization.pr_clip import RenderRequest, output_directory, render
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathRole,
)
from src.utils.paths import PROJECT_ROOT

PATH_BOUNDARY = NonHydraPathBoundary(
    name="slcs.render_pr_clip",
    fields=(
        BoundaryPathField(
            "overlay",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.ANY,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "scene",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.ANY,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "output_root",
            PathRole.OUTPUT,
            PathDirection.OUTPUT,
            PathKind.ANY,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY
        ),
    ),
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("overlay", "scene"):
        result.add_argument(f"--{name}", type=Path, required=True)
    for name in (
        "experiment",
        "run-id",
        "label",
        "model",
        "clip-id",
        "camera-id",
        "checkpoint-sha256",
    ):
        result.add_argument(f"--{name}", required=True)
    result.add_argument("--epoch", type=int, required=True)
    result.add_argument("--start", type=float, required=True)
    result.add_argument("--end", type=float, required=True)
    result.add_argument("--fps", type=float, default=10)
    result.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "slcs" / "visualize",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    request = RenderRequest(**vars(args))
    PATH_BOUNDARY.validate(
        {
            "overlay": args.overlay,
            "scene": args.scene,
            "output_root": args.output_root,
            "output": output_directory(request),
        },
        resolver=cli_resolver(args.output_root),
        independent_artifact_inputs=True,
    )
    print(render(request, command_line=tuple(sys.argv)))


if __name__ == "__main__":
    main()
