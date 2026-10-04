"""Read-only browser review of real saved Ball Refiner 2D datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import uvicorn

from src.tasks.ball_refiner.refiner_2d.review.artifacts import ReviewArtifacts
from src.tasks.ball_refiner.refiner_2d.review.web import create_app
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
    name="ball_refiner.review_2d_dataset",
    fields=(
        BoundaryPathField("evidence", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.DIRECTORY,
                          must_exist=True, allow_role_root=True),
        BoundaryPathField("predictions", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.DIRECTORY,
                          must_exist=True, allow_role_root=True, required=False),
        BoundaryPathField("context_root", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.DIRECTORY,
                          must_exist=True, allow_role_root=True, required=False),
        BoundaryPathField("rgb_store", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.DIRECTORY,
                          must_exist=True, allow_role_root=True, required=False),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, help="Complete saved cached-comparison GMM output")
    parser.add_argument("--context-root", type=Path, help="Complete context cache or directory of complete clip shards")
    parser.add_argument("--rgb-store", type=Path, help="Explicit image provider; exact JPEG/frame/PTS/geometry must match")
    parser.add_argument("--pose-threshold", type=float, default=0.5)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8894)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must lie in [1,65535]")
    root = args.evidence.parent
    roots = RuntimePathRoots(project_root=root, data_root=root, artifact_root=root, output_root=root,
                             checkpoint_root=root, cache_root=root, external_asset_root=root)
    supplied = {name: getattr(args, name) for name in ("evidence", "predictions", "context_root", "rgb_store")
                if getattr(args, name) is not None}
    paths = PATH_BOUNDARY.validate(supplied, resolver=PathResolver(roots), independent_artifact_inputs=True)
    optional: dict[str, Path | None] = {name: None for name in ("predictions", "context_root", "rgb_store")}
    for name in optional:
        if name in supplied:
            optional[name] = paths.declared(name).path
    torch.set_num_threads(2)
    artifacts = ReviewArtifacts(paths.declared("evidence").path, predictions_root=optional["predictions"],
                               context_root=optional["context_root"], rgb_store=optional["rgb_store"],
                               pose_threshold=args.pose_threshold)
    print(f"Ball Refiner 2D dataset review · http://{args.host}:{args.port}", flush=True)
    uvicorn.run(create_app(artifacts), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
