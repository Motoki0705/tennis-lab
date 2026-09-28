"""CPU-only label and indexed-context audit; see the task README."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_refiner.data.audit import audit_store
from src.tennis_scene.pipeline.artifacts import write_json_atomic
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
    name="ball_refiner.audit_data",
    fields=(
        BoundaryPathField("store", PathRole.DATA, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True),
        BoundaryPathField("context", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True),
        BoundaryPathField("output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--meiji-context-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New absolute ball_refiner/analyze/<experiment>/<run> directory")
    parser.add_argument("--pose-threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not all(path.is_absolute() for path in (args.store, args.meiji_context_root, args.output)):
        parser.error("All paths must be absolute")
    if not math.isfinite(args.pose_threshold) or not 0 <= args.pose_threshold <= 1:
        parser.error("--pose-threshold must be finite and in [0,1]")
    roots = RuntimePathRoots(
        project_root=args.output.parent, data_root=args.store.parent, artifact_root=args.meiji_context_root.parent,
        output_root=args.output.parent, checkpoint_root=args.output.parent,
        cache_root=args.output.parent, external_asset_root=args.output.parent,
    )
    paths = PATH_BOUNDARY.validate(
        {"store": args.store, "context": args.meiji_context_root, "output": args.output},
        resolver=PathResolver(roots),
    )
    store, context, output = (paths.declared(name).path for name in ("store", "context", "output"))
    if output.exists():
        raise FileExistsError(f"Audit output already exists: {output}")
    result = audit_store(BallFrameStore(store), meiji_context_root=context,
                         pose_threshold=args.pose_threshold)
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "audit.json", result)
    print(json.dumps({key: result[key] for key in ("counts", "context_counts")}, indent=2))


if __name__ == "__main__":
    main()
