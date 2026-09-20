"""Enqueue the complete real RGB dataset recipe, or one selected source."""

import argparse

from src.tennis_scene.dataset_pipeline.orchestration import run_build
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)
from src.utils.paths import PROJECT_ROOT

PATH_BOUNDARY = NonHydraPathBoundary(
    name="tennis_scene.build_real_rgb",
    fields=(
        BoundaryPathField(
            "project",
            PathRole.PROJECT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "mode", choices=("all", "broadcast", "meiji"), nargs="?", default="all"
    )
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    roots = RuntimePathRoots(
        **{f"{role.value}_root": PROJECT_ROOT for role in PathRole}
    )
    paths = PATH_BOUNDARY.validate(
        {"project": PROJECT_ROOT}, resolver=PathResolver(roots)
    )
    run_build(
        args.mode,
        args.overrides,
        execute=args.execute,
        project=paths.declared("project").path,
    )


if __name__ == "__main__":
    main()
