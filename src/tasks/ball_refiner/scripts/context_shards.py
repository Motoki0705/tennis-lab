"""Plan, generate one whole clip, or assemble all context shards; CUDA via queue."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_refiner.data.context_shards import (
    generate_context_shard,
    merge_context_shards,
    write_context_plan,
)
from src.tasks.ball_refiner.data.evidence_cache import EvidenceCache
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
    name="ball_refiner.context_shards",
    fields=(
        BoundaryPathField("store", PathRole.DATA, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True),
        BoundaryPathField("evidence", PathRole.CACHE, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True),
        BoundaryPathField("output", PathRole.CACHE, PathDirection.OUTPUT, PathKind.DIRECTORY),
        BoundaryPathField("scene_config", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE, must_exist=True, required=False),
        BoundaryPathField("plan", PathRole.CACHE, PathDirection.INPUT, PathKind.FILE, must_exist=True, required=False),
        BoundaryPathField("shard", PathRole.CACHE, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True, many=True, required=False),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "generate", "merge"))
    for name in ("store", "evidence", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--scene-config", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--shard", type=Path, action="append", help="Exact complete shard paths, including explicit replacement attempts")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--max-tracks", type=int)
    args = parser.parse_args()
    required = {
        "plan": {"scene_config", "max_tracks"},
        "generate": {"scene_config", "max_tracks", "plan", "shard_index"},
        "merge": {"plan", "shard"},
    }[args.action]
    supplied = {name for name in ("scene_config", "max_tracks", "plan", "shard_index", "shard")
                if getattr(args, name) is not None}
    if supplied != required:
        parser.error(f"{args.action} requires exactly these additional arguments: {sorted(required)}")
    path_names = ("store", "evidence", "output", "scene_config", "plan")
    arguments = {name: getattr(args, name) for name in path_names if getattr(args, name) is not None}
    all_paths = [*arguments.values(), *(args.shard or [])]
    if not all(path.is_absolute() for path in all_paths):
        parser.error("All paths must be absolute")
    if args.shard is not None:
        arguments["shard"] = args.shard
    cache_paths = [args.evidence, args.output, *(args.shard or [])]
    if args.plan is not None:
        cache_paths.append(args.plan)
    artifact_root = args.scene_config.parent if args.scene_config is not None else args.output.parent
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT, data_root=args.store.parent, checkpoint_root=artifact_root,
        cache_root=Path(os.path.commonpath([path.parent for path in cache_paths])),
        artifact_root=artifact_root, output_root=args.output.parent, external_asset_root=PROJECT_ROOT / "third_party",
    )
    paths = PATH_BOUNDARY.validate(arguments, resolver=PathResolver(roots))
    output = paths.declared("output").path
    if output.exists():
        raise FileExistsError(f"Output must be new: {output}")
    evidence = EvidenceCache(paths.declared("evidence").path, BallFrameStore(paths.declared("store").path))
    if args.action == "merge":
        result = merge_context_shards(
            evidence, plan_path=paths.declared("plan").path, output=output,
            shards=tuple(item.path for item in paths.declared_many("shard")),
        )
    else:
        # CPU assembly does not import or load model runtimes or compiled ops.
        from src.tasks.ball_refiner.data.context_inference import load_context_producer

        producer = load_context_producer(paths.declared("scene_config").path, max_tracks=args.max_tracks)
        if args.action == "plan":
            result = write_context_plan(evidence, producer=producer, output=output)
        else:
            result = generate_context_shard(
                evidence, plan_path=paths.declared("plan").path, shard_index=args.shard_index,
                producer=producer, output=output,
            )
    print(result)


if __name__ == "__main__":
    main()
