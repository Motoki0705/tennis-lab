"""Review saved cross-camera boxes, partial labels and optional historical scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import uvicorn

from src.tasks.player_association.review.service import AssociationReviewService
from src.tasks.player_association.review.web import create_app
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
    name="player_association.dataset_review",
    fields=(
        BoundaryPathField(
            "dataset_root",
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
            "sides",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
            required=False,
        ),
        BoundaryPathField(
            "score_reports",
            PathRole.ARTIFACT,
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
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument(
        "--sides",
        type=Path,
        help="Artifact-root-relative saved reviewed-ball side report",
    )
    parser.add_argument(
        "--score-report",
        action="append",
        type=Path,
        default=[],
        help="Artifact-root-relative historical evaluate.json; repeat to compare",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8897)
    args = parser.parse_args()
    dataset_root, artifact_root = (
        args.dataset_root.expanduser().resolve(),
        args.artifact_root.expanduser().resolve(),
    )
    roots = RuntimePathRoots(
        project_root=dataset_root.parent,
        data_root=dataset_root,
        artifact_root=artifact_root,
        checkpoint_root=artifact_root,
        output_root=artifact_root,
        cache_root=artifact_root,
        external_asset_root=artifact_root,
    )
    resolver = PathResolver(roots)
    sides = (
        None if args.sides is None else resolver.resolve(PathRole.ARTIFACT, args.sides)
    )
    reports = tuple(
        resolver.resolve(PathRole.ARTIFACT, path) for path in args.score_report
    )
    inputs: dict[str, object] = {
        "dataset_root": dataset_root,
        "artifact_root": artifact_root,
    }
    if sides is not None:
        inputs["sides"] = sides
    if reports:
        inputs["score_reports"] = reports
    paths = PATH_BOUNDARY.validate(inputs, resolver=resolver)
    cv2.setNumThreads(1)
    service = AssociationReviewService(
        paths.declared("dataset_root").path,
        paths.declared("artifact_root").path,
        sides=sides,
        score_reports=reports,
    )
    print(f"Player Association review · http://{args.host}:{args.port}", flush=True)
    uvicorn.run(create_app(service), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
