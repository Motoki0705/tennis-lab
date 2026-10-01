"""Plan or resume Meiji-only COCO17 context. Generation must run in the queue."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
from pathlib import Path
from typing import Any

import cv2
import torch

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_refiner.data.evidence_cache import EvidenceCache
from src.tasks.ball_refiner.data.meiji_context import (
    resume_meiji_cache,
    selection,
    write_meiji_plan,
)
from src.tasks.ball_refiner.data.meiji_context_inference import load_meiji_producer
from src.utils.checksum import dual_sha256
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

PATH_BOUNDARY = NonHydraPathBoundary(name="ball_refiner.meiji_context", fields=(
    BoundaryPathField("scene_config", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
    BoundaryPathField("store", PathRole.DATA, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True),
    BoundaryPathField("evidence", PathRole.CACHE, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True),
    BoundaryPathField("output", PathRole.CACHE, PathDirection.OUTPUT, PathKind.DIRECTORY),
    BoundaryPathField("freeze", PathRole.PROJECT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "generate"))
    for name in ("store", "evidence", "scene-config", "freeze", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--plan-sha256")
    args = parser.parse_args()
    if not all(getattr(args, k).is_absolute() for k in ("store", "evidence", "scene_config", "freeze", "output")):
        parser.error("All paths must be absolute")
    if (args.action == "generate") != (args.plan_sha256 is not None):
        parser.error("Only generate requires --plan-sha256")
    roots = RuntimePathRoots(project_root=PROJECT_ROOT, data_root=args.store.parent,
        checkpoint_root=args.scene_config.parent, artifact_root=args.scene_config.parent,
        cache_root=Path(os.path.commonpath([args.evidence.parent, args.output.parent])),
        output_root=args.output.parent, external_asset_root=PROJECT_ROOT / "third_party")
    PATH_BOUNDARY.validate({k: getattr(args, k) for k in ("store", "evidence", "scene_config", "freeze", "output")},
                           resolver=PathResolver(roots))
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    evidence = EvidenceCache(args.evidence, BallFrameStore(args.store))
    names, _ = selection(evidence)
    if len(names) != 108 or sum(evidence.store.clip_by_id(n).frame_count for n in names) != 52866:
        raise ValueError("Require exactly the approved 108 Meiji camera-clips / 52,866 frames")
    producer = load_meiji_producer(args.scene_config, args.freeze)
    if args.action == "plan":
        path = write_meiji_plan(evidence, producer, args.output)
        print(json.dumps({"plan": str(path), "sha256": dual_sha256(path)}), flush=True)
        return
    if producer.people.runtime.device != "cuda":
        raise ValueError("The approved cache job requires explicit cuda, without device fallback")
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(7 * 1024**3 / torch.cuda.get_device_properties(0).total_memory, 0)

    def interrupted(signum: int, frame: Any) -> None:
        raise RuntimeError(f"Context generation interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    with (args.output / "generation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(resume_meiji_cache(evidence, producer, args.output / "plan.json", args.plan_sha256), flush=True)


if __name__ == "__main__":
    main()
