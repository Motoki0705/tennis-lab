"""Create an independently readable, lossless compressed Court dataset copy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.synthetic_data_generation.dataset.court.storage import compact
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)

PATH_BOUNDARY = NonHydraPathBoundary(
    name="synthetic.court_storage_compaction",
    fields=(
        BoundaryPathField(
            "data_root",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "output_root",
            PathRole.OUTPUT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "source",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
        ),
        BoundaryPathField(
            "destination",
            PathRole.OUTPUT,
            PathDirection.OUTPUT,
            PathKind.DIRECTORY,
        ),
        BoundaryPathField(
            "report",
            PathRole.OUTPUT,
            PathDirection.OUTPUT,
            PathKind.FILE,
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not args.data_root.is_absolute() or not args.output_root.is_absolute():
        parser.error("--data-root and --output-root must be absolute paths")
    data_root = args.data_root.resolve(strict=False)
    output_root = args.output_root.resolve(strict=False)
    project_root = data_root.parent
    roots = RuntimePathRoots(
        project_root=project_root,
        data_root=data_root,
        checkpoint_root=output_root,
        artifact_root=output_root,
        output_root=output_root,
        cache_root=output_root,
        external_asset_root=data_root,
    )
    paths = PATH_BOUNDARY.validate(
        {
            "data_root": data_root,
            "output_root": output_root,
            "source": args.source,
            "destination": args.destination,
            "report": args.report,
        },
        resolver=PathResolver(roots),
    )
    result = compact(
        paths.declared("source").path,
        paths.declared("destination").path,
        resume=args.resume,
        workers=args.workers,
    )
    report = paths.declared("report").path
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
