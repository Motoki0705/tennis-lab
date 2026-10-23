"""Browse published Court trajectories and image-label overlays locally."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from src.synthetic_data_generation.dataset.court.review.scale_service import (
    ScaleReviewService,
)
from src.synthetic_data_generation.dataset.court.review.service import ReviewService
from src.synthetic_data_generation.dataset.court.review.web import create_app
from src.synthetic_data_generation.dataset.court.scale.publication import read_json
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
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--scenes-root", type=Path)
    parser.add_argument("--dataset-version", type=Path, nargs="+")
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()

    if args.dataset_version or args.catalog:
        if args.dataset_version and args.catalog or args.data_root or args.scenes_root:
            parser.error(
                "Use dataset-version/catalog or data-root + scenes-root exclusively."
            )
        versions = args.dataset_version
        if args.catalog:
            catalog = read_json(args.catalog)
            if catalog["schema"] != "court_scale_review_catalog_v1":
                raise ValueError("Unknown Court scale Review catalog schema.")
            versions = [
                args.catalog.resolve().parent / p for p in catalog["dataset_versions"]
            ]
        service: ReviewService = ScaleReviewService(versions)
        print(f"Court Review · http://127.0.0.1:{args.port}", flush=True)
        uvicorn.run(create_app(service), host="127.0.0.1", port=args.port)
        return
    if args.data_root is None or args.scenes_root is None:
        parser.error("Legacy Review requires both --data-root and --scenes-root.")

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
