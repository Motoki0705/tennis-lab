"""Review saved synthetic 3D ball datasets without generation or inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from src.tasks.ball_refiner.refiner_3d.review.data import DatasetReview
from src.tasks.ball_refiner.refiner_3d.review.web import create_app
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
    name="ball_refiner.dataset_3d_review",
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
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Absolute directory containing dataset folders (e.g. data/ball_refiner)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8893)
    args = parser.parse_args()
    if not args.data_root.is_absolute():
        parser.error("--data-root must be absolute")
    data_root = args.data_root.resolve()
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT,
        data_root=data_root,
        artifact_root=data_root,
        output_root=data_root,
        checkpoint_root=data_root,
        cache_root=data_root,
        external_asset_root=data_root,
    )
    paths = PATH_BOUNDARY.validate(
        {"data_root": data_root}, resolver=PathResolver(roots)
    )
    service = DatasetReview(paths.declared("data_root").path)
    print(f"Ball Refiner 3D dataset review: http://{args.host}:{args.port}", flush=True)
    uvicorn.run(create_app(service), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
