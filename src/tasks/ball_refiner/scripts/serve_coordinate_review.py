"""Serve synchronized shared-dataset, augmentation and 2D/3D model review."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import uvicorn

from src.tasks.ball_refiner.coordinates.review.service import ReviewService
from src.tasks.ball_refiner.coordinates.review.web import create_app
from src.tasks.base.visualization.inference_queue import shared_repository_root
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
    name="ball_refiner.serve_coordinate_review",
    fields=(
        BoundaryPathField("data_root", PathRole.DATA, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True, allow_role_root=True),
        BoundaryPathField("outputs_root", PathRole.OUTPUT, PathDirection.INPUT, PathKind.DIRECTORY, allow_role_root=True),
        BoundaryPathField("checkpoints_root", PathRole.CHECKPOINT, PathDirection.INPUT, PathKind.DIRECTORY, allow_role_root=True),
    ),
)


def main() -> None:
    root = shared_repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root / "data/ball_refiner/single_object")
    parser.add_argument("--outputs-root", type=Path, default=root / "outputs/ball_refiner")
    parser.add_argument("--checkpoints-root", type=Path, default=root / "ckpt/ball_refiner")
    parser.add_argument("--port", type=int, default=8784)
    parser.add_argument("--host", choices=("127.0.0.1", "localhost", "::1"), default="127.0.0.1")
    args = parser.parse_args()
    values = {name: getattr(args, name) for name in ("data_root", "outputs_root", "checkpoints_root")}
    if not all(value.is_absolute() for value in values.values()):
        parser.error("root arguments must be explicit absolute paths")
    roots = RuntimePathRoots(project_root=root, data_root=args.data_root, output_root=args.outputs_root,
                             checkpoint_root=args.checkpoints_root, artifact_root=root / "outputs",
                             cache_root=root / ".cache", external_asset_root=root / "third_party")
    paths = PATH_BOUNDARY.validate(values, resolver=PathResolver(roots))
    torch.set_num_threads(2)
    service = ReviewService(**{name: paths.declared(name).path for name in values})
    uvicorn.run(create_app(service), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
