"""Browse published Court trajectories and image-label overlays locally."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from src.synthetic_data_generation.dataset.court.review.service import ReviewService
from src.synthetic_data_generation.dataset.court.review.web import create_app
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
    name="synthetic.court_review",
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
            "scenes_root",
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
    parser.add_argument("--scenes-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
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
        {"data_root": data_root, "scenes_root": args.scenes_root},
        resolver=PathResolver(roots),
    )
    service = ReviewService(paths.declared("scenes_root").path)
    print(f"Court Review · http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(create_app(service), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
