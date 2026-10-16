"""Serve synchronized shared-dataset, augmentation and 3D trajectory and event review."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import uvicorn

from src.tasks.ball_refiner_3d.configuration.visualization import (
    ReviewServerConfig,
    review_server_config,
)
from src.tasks.ball_refiner_3d.visualization.dataset_review.artifacts import cache_root
from src.tasks.ball_refiner_3d.visualization.dataset_review.service import ReviewService
from src.tasks.ball_refiner_3d.visualization.dataset_review.web import create_app
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
    name="ball_refiner_3d.review_dataset",
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
            "outputs_root",
            PathRole.OUTPUT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "checkpoints_root",
            PathRole.CHECKPOINT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "prediction_cache_root",
            PathRole.OUTPUT,
            PathDirection.OUTPUT,
            PathKind.DIRECTORY,
        ),
    ),
)


def main() -> None:
    root = shared_repository_root()
    defaults = review_server_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=root / "data/ball_refiner/single_object"
    )
    parser.add_argument(
        "--outputs-root", type=Path, default=root / "outputs/ball_refiner_3d"
    )
    parser.add_argument(
        "--checkpoints-root", type=Path, default=root / "ckpt/ball_refiner_3d"
    )
    parser.add_argument("--port", type=int, default=defaults.port)
    parser.add_argument(
        "--host", choices=("127.0.0.1", "localhost", "::1"), default=defaults.host
    )
    parser.add_argument(
        "--prepare-saved-predictions",
        action="store_true",
        help="Recompute best checkpoints' test predictions on CPU into a separate review cache, then exit",
    )
    parser.add_argument("--cpu-threads", type=int, default=defaults.cpu_threads)
    args = parser.parse_args()
    server = ReviewServerConfig(args.host, args.port, args.cpu_threads)
    values = {
        name: getattr(args, name)
        for name in ("data_root", "outputs_root", "checkpoints_root")
    }
    if not all(value.is_absolute() for value in values.values()):
        parser.error("root arguments must be explicit absolute paths")
    roots = RuntimePathRoots(
        project_root=root,
        data_root=args.data_root,
        output_root=args.outputs_root,
        checkpoint_root=args.checkpoints_root,
        artifact_root=root / "outputs",
        cache_root=root / ".cache",
        external_asset_root=root / "third_party",
    )
    paths = PATH_BOUNDARY.validate(
        {**values, "prediction_cache_root": cache_root(args.outputs_root)},
        resolver=PathResolver(roots),
    )
    torch.set_num_threads(server.cpu_threads)
    service = ReviewService(**{name: paths.declared(name).path for name in values})
    if args.prepare_saved_predictions:
        from src.tasks.ball_refiner_3d.visualization.dataset_review.prepare import (
            prepare_saved_predictions,
        )

        prepare_saved_predictions(service)
        return
    uvicorn.run(create_app(service), host=server.host, port=server.port)


if __name__ == "__main__":
    main()
