"""Browse raw ACCAD mocap in the PLCS world coordinate system, locally."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from src.tasks.plcs.visualization.review.service import ReviewService
from src.tasks.plcs.visualization.review.web import create_app
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
    name="plcs.accad_motion_review",
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
            "accad_root",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
        ),
        BoundaryPathField(
            "smplh_root",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--accad-root",
        type=Path,
        default=None,
        help="Defaults to <data-root>/ACCAD.",
    )
    parser.add_argument(
        "--smplh-root",
        type=Path,
        default=None,
        help="Directory holding <gender>/model.npz. Defaults to <data-root>/smplh.",
    )
    parser.add_argument("--port", type=int, default=8769)
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    accad_root = (
        args.accad_root.expanduser() if args.accad_root else data_root / "ACCAD"
    ).resolve()
    smplh_root = (
        args.smplh_root.expanduser() if args.smplh_root else data_root / "smplh"
    ).resolve()
    roots = RuntimePathRoots(
        project_root=data_root.parent,
        data_root=data_root,
        checkpoint_root=data_root,
        artifact_root=data_root,
        output_root=data_root,
        cache_root=data_root,
        external_asset_root=data_root,
    )
    paths = PATH_BOUNDARY.validate(
        {
            "data_root": data_root,
            "accad_root": accad_root,
            "smplh_root": smplh_root,
        },
        resolver=PathResolver(roots),
    )
    service = ReviewService(
        paths.declared("accad_root").path,
        paths.declared("smplh_root").path,
    )
    print(f"ACCAD Review · http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(create_app(service), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
