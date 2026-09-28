"""CPU-only label and indexed-context audit; see the task README."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_refiner.data.audit import audit_store
from src.tennis_scene.pipeline.artifacts import write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--meiji-context-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New absolute ball_refiner/analyze/<experiment>/<run> directory")
    parser.add_argument("--pose-threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not all(path.is_absolute() for path in (args.store, args.meiji_context_root, args.output)):
        parser.error("All paths must be absolute")
    if args.output.exists():
        raise FileExistsError(f"Audit output already exists: {args.output}")
    result = audit_store(BallFrameStore(args.store), meiji_context_root=args.meiji_context_root,
                         pose_threshold=args.pose_threshold)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(args.output / "audit.json", result)
    print(json.dumps({key: result[key] for key in ("counts", "context_counts")}, indent=2))


if __name__ == "__main__":
    main()
