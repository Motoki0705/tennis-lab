"""Browse saved tracking, ID intervals and explicit missing states without inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import uvicorn

from src.tasks.person_tracking.review.service import TrackingReviewService
from src.tasks.person_tracking.review.web import create_app
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
    name="person_tracking.dataset_review",
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
            "artifact_root",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "campaign",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            required=False,
        ),
        BoundaryPathField(
            "pose_dataset",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            required=False,
        ),
        BoundaryPathField(
            "stores",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            many=True,
            required=False,
        ),
        BoundaryPathField(
            "references",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
            many=True,
            required=False,
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--pose-dataset", type=Path)
    parser.add_argument("--store", type=Path, action="append")
    parser.add_argument(
        "--reference",
        type=Path,
        action="append",
        help="Partial box reference labels for the supplied component clips",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8896)
    args = parser.parse_args()
    # Validate absolute roots before resolving: relative paths never gain CWD authority.
    if not args.data_root.is_absolute() or not args.artifact_root.is_absolute():
        parser.error("--data-root and --artifact-root must be absolute")
    roots = RuntimePathRoots(
        project_root=args.artifact_root.resolve(),
        data_root=args.data_root.resolve(),
        artifact_root=args.artifact_root.resolve(),
        checkpoint_root=args.artifact_root.resolve(),
        output_root=args.artifact_root.resolve(),
        cache_root=args.artifact_root.resolve(),
        external_asset_root=args.data_root.resolve(),
    )
    arguments: dict[str, object] = {
        "data_root": args.data_root,
        "artifact_root": args.artifact_root,
    }
    if args.campaign is not None:
        arguments["campaign"] = args.campaign
    if args.pose_dataset is not None:
        arguments["pose_dataset"] = args.pose_dataset
    if args.store is not None:
        arguments["stores"] = args.store
    if args.reference is not None:
        arguments["references"] = args.reference
    resolver = PathResolver(roots)
    paths = PATH_BOUNDARY.validate(arguments, resolver=resolver)
    service = TrackingReviewService(
        resolver=resolver,
        campaign=None if "campaign" not in paths else paths.declared("campaign").path,
        pose_dataset=None
        if "pose_dataset" not in paths
        else paths.declared("pose_dataset").path,
        stores=()
        if "stores" not in paths
        else tuple(item.path for item in paths.declared_many("stores")),
        reference_files=()
        if "references" not in paths
        else tuple(item.path for item in paths.declared_many("references")),
    )
    cv2.setNumThreads(1)
    print(
        f"Person Tracking dataset review · http://{args.host}:{args.port}", flush=True
    )
    uvicorn.run(create_app(service), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
