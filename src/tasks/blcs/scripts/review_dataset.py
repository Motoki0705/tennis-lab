"""Browse generated BLCS dataset scenes in 3D, locally."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from src.tasks.blcs.visualization.review.dataset_service import (
    BLCSDatasetReviewService,
)
from src.tasks.blcs.visualization.review.web import create_dataset_app
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
    name="blcs.dataset_scene_review",
    fields=(
        BoundaryPathField(
            "data_root",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--form",
        action="append",
        default=None,
        help="Limit the catalog to one form; repeat for several.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8773)
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
        {"data_root": data_root},
        resolver=PathResolver(roots),
    )
    service = BLCSDatasetReviewService(
        paths.declared("data_root").path,
        forms=args.form,
    )
    print(f"BLCS dataset review · http://{args.host}:{args.port}", flush=True)
    uvicorn.run(create_dataset_app(service), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
