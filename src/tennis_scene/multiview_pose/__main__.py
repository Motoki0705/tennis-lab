"""Run framewise 3D pose on saved, synchronized all-person observations."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tennis_scene.multiview_pose.inputs import PinnedCourtSide, read_saved_inputs
from src.tennis_scene.multiview_pose.reconstruct import DEFAULT_CONFIG, reconstruct
from src.tennis_scene.multiview_pose.storage import publish_candidates
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathContractError,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)
from src.utils.paths import PROJECT_ROOT

PATH_BOUNDARY = NonHydraPathBoundary(
    name="tennis_scene.multiview_pose",
    fields=(
        BoundaryPathField(
            "clip_store",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "side_store",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "court_store",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "drift_store",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
            required=False,
        ),
        BoundaryPathField(
            "output",
            PathRole.OUTPUT,
            PathDirection.OUTPUT,
            PathKind.DIRECTORY,
            allow_role_root=True,
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clip-store",
        required=True,
        type=Path,
        help="Saved DINO, ViTPose and local calibration store",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Separate output clip store"
    )
    parser.add_argument(
        "--side-store",
        required=True,
        type=Path,
        help="Ball-based video court side store",
    )
    parser.add_argument(
        "--side-digest", required=True, help="Pinned side decision SHA-256"
    )
    parser.add_argument(
        "--court-store",
        required=True,
        type=Path,
        help="Shared court store named by the pinned side",
    )
    parser.add_argument(
        "--drift-store",
        type=Path,
        help="Drift store when the pinned decision names drift evidence",
    )
    args = parser.parse_args()
    requested_output = args.output.expanduser()
    arguments = {
        "clip_store": args.clip_store.expanduser().resolve(),
        "side_store": args.side_store.expanduser().resolve(),
        "court_store": args.court_store.expanduser().resolve(),
        "output": requested_output.resolve(),
    }
    if args.drift_store is not None:
        arguments["drift_store"] = args.drift_store.expanduser().resolve()
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT,
        data_root=PROJECT_ROOT,
        checkpoint_root=PROJECT_ROOT,
        artifact_root=PROJECT_ROOT,
        output_root=arguments["output"],
        cache_root=PROJECT_ROOT,
        external_asset_root=PROJECT_ROOT,
    )
    paths = PATH_BOUNDARY.validate(
        arguments, resolver=PathResolver(roots), independent_artifact_inputs=True
    )
    clip_store = paths.declared("clip_store").path
    output = paths.declared("output").path
    if output.exists() and not output.is_dir():
        raise PathContractError(f"Pose output must be a directory: {output}")
    drift_store = paths.declared("drift_store").path if "drift_store" in paths else None
    side = PinnedCourtSide.load(
        paths.declared("side_store").path,
        args.side_digest,
        paths.declared("court_store").path,
        drift_store,
    )
    inputs = read_saved_inputs(clip_store, side)
    result = reconstruct(inputs.observations, DEFAULT_CONFIG)
    ref = publish_candidates(output, clip_store, inputs, result, DEFAULT_CONFIG)
    print(requested_output / ref.path)


if __name__ == "__main__":
    main()
