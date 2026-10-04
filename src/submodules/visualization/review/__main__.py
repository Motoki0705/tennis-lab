"""Launch a CPU-only review of an explicit saved component store."""

from __future__ import annotations

import argparse
from pathlib import Path

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
    name="submodules.pose_review",
    fields=(
        BoundaryPathField("store", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True, allow_role_root=True),
        BoundaryPathField("topology", PathRole.CHECKPOINT, PathDirection.INPUT, PathKind.FILE, required=False, must_exist=True),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True, help="Saved component store containing scene.json")
    parser.add_argument("--topology", type=Path, help="Optional SMPL-H npz: only its triangle array f is read; no model is run")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8902)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be in 1..65535")
    if not args.store.is_absolute() or (args.topology is not None and not args.topology.is_absolute()):
        parser.error("store and topology must be explicit absolute paths")
    store = args.store.expanduser().resolve()
    topology = args.topology.expanduser().resolve() if args.topology is not None else None
    checkpoint_root = topology.parent if topology is not None else store
    roots = RuntimePathRoots(project_root=store, data_root=store, checkpoint_root=checkpoint_root, artifact_root=store, output_root=store, cache_root=store, external_asset_root=store)
    values = {"store": store}
    if topology is not None:
        values["topology"] = topology
    paths = PATH_BOUNDARY.validate(values, resolver=PathResolver(roots))
    import uvicorn

    from src.submodules.visualization.review.store import PoseReviewStore
    from src.submodules.visualization.review.web import create_app

    topology_path = paths.declared("topology").path if "topology" in paths else None
    snapshot = PoseReviewStore(paths.declared("store").path, topology=topology_path)
    uvicorn.run(create_app(snapshot), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
