"""Create matched quantitative tables and same-image qualitative comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tasks.court_detection.ablation.comparison import compare_evaluations
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
    name="court_detection.ablation_comparison",
    fields=(
        BoundaryPathField(
            "evaluations",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
            allow_role_root=True,
            many=True,
        ),
        BoundaryPathField(
            "output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", nargs=4, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT,
        data_root=PROJECT_ROOT,
        checkpoint_root=PROJECT_ROOT,
        artifact_root=PROJECT_ROOT,
        output_root=output.parent,
        cache_root=PROJECT_ROOT,
        external_asset_root=PROJECT_ROOT,
    )
    paths = PATH_BOUNDARY.validate(
        {
            "evaluations": [path.expanduser().resolve() for path in args.evaluations],
            "output": output,
        },
        resolver=PathResolver(roots),
        independent_artifact_inputs=True,
    )
    print(
        compare_evaluations(
            [entry.path for entry in paths.declared_many("evaluations")],
            paths.declared("output").path,
        )
    )


if __name__ == "__main__":
    main()
