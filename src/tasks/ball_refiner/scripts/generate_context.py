"""Generate JPEG-aligned context; CUDA execution must use the shared queue."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_refiner.data.context_cache import generate_context_cache
from src.tasks.ball_refiner.data.context_inference import load_context_producer
from src.tasks.ball_refiner.data.evidence_cache import EvidenceCache
from src.tennis_scene.pipeline.artifacts import document_digest
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
    name="ball_refiner.generate_context",
    fields=(
        BoundaryPathField("scene_config", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
        BoundaryPathField("store", PathRole.DATA, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True),
        BoundaryPathField("evidence", PathRole.CACHE, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True),
        BoundaryPathField("output", PathRole.CACHE, PathDirection.OUTPUT, PathKind.DIRECTORY),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scene-config", "store", "evidence", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--max-tracks", type=int, required=True)
    parser.add_argument("--clip-id", action="append", help="Explicit pilot subset; omit to generate all detector clips")
    parser.add_argument("--dry-run", action="store_true", help="CPU asset/selection preflight; never loads a model")
    args = parser.parse_args()
    names = ("scene_config", "store", "evidence", "output")
    if not all(getattr(args, name).is_absolute() for name in names):
        parser.error("All paths must be absolute")
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT, data_root=args.store.parent, checkpoint_root=args.scene_config.parent,
        cache_root=Path(os.path.commonpath([args.evidence.parent, args.output.parent])),
        artifact_root=args.scene_config.parent, output_root=args.output.parent,
        external_asset_root=PROJECT_ROOT / "third_party",
    )
    paths = PATH_BOUNDARY.validate({name: getattr(args, name) for name in names}, resolver=PathResolver(roots))
    producer = load_context_producer(paths.declared("scene_config").path, max_tracks=args.max_tracks)
    evidence = EvidenceCache(paths.declared("evidence").path, BallFrameStore(paths.declared("store").path))
    selected = evidence.clip_ids if args.clip_id is None else tuple(args.clip_id)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(evidence.clip_ids):
        raise ValueError("Select unique clips from the detector cache")
    if paths.declared("output").path.exists():
        raise FileExistsError("Context output must be new, including during preflight")
    if args.dry_run:
        identity = producer.identity()
        print(json.dumps({"status": "preflight_only", "model_identity_sha256": document_digest(identity),
                          "assets": identity["assets"], "max_tracks": args.max_tracks,
                          "clips": [{"id": name, "frames": evidence.store.clip_by_id(name).frame_count,
                                     "source": evidence.store.clip_by_id(name).source} for name in selected]}))
        return
    generate_context_cache(evidence, output=paths.declared("output").path, producer=producer,
                           clip_ids=None if args.clip_id is None else selected)


if __name__ == "__main__":
    main()
