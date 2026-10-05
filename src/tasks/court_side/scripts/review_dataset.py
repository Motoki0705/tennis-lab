"""Review saved multi-view stores and optional production frame diagnostics locally."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from src.tasks.court_side.review.service import ReviewService
from src.tasks.court_side.review.web import create_app
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
    name="court_side.dataset_review",
    fields=(
        BoundaryPathField(
            "store",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "diagnostics",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            required=False,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "comparison_store",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            required=False,
            must_exist=True,
            allow_role_root=True,
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store",
        type=Path,
        required=True,
        help="Published component store containing scene.json",
    )
    parser.add_argument(
        "--diagnostics",
        type=Path,
        help="Saved production.json, production-frames.csv and production-observations.npz",
    )
    parser.add_argument(
        "--comparison-store",
        type=Path,
        help="Optional independent store, shown as a separate case",
    )
    parser.add_argument("--port", type=int, default=8899)
    args = parser.parse_args()
    store = args.store.expanduser().resolve()
    roots = RuntimePathRoots(
        project_root=store,
        data_root=store,
        checkpoint_root=store,
        artifact_root=store,
        output_root=store,
        cache_root=store,
        external_asset_root=store,
    )
    diagnostics = (
        None if args.diagnostics is None else args.diagnostics.expanduser().resolve()
    )
    comparison = (
        None
        if args.comparison_store is None
        else args.comparison_store.expanduser().resolve()
    )
    arguments = {"store": store}
    if diagnostics is not None:
        arguments["diagnostics"] = diagnostics
    if comparison is not None:
        arguments["comparison_store"] = comparison
    paths = PATH_BOUNDARY.validate(
        arguments, resolver=PathResolver(roots), independent_artifact_inputs=True
    )
    service = ReviewService(
        paths.declared("store").path,
        diagnostics=None if diagnostics is None else paths.declared("diagnostics").path,
        comparison_stores=()
        if comparison is None
        else (paths.declared("comparison_store").path,),
    )
    print(
        f"Court Side review: http://127.0.0.1:{args.port} · saved artifacts only",
        flush=True,
    )
    uvicorn.run(create_app(service), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
