"""Generate frozen detector evidence. CUDA execution must use the training queue."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tasks.ball_detection.data.store import SOURCES, SPLIT_CODES
from src.tasks.ball_detection.model_io.contracts import BallCandidateConfig
from src.tasks.ball_refiner.data.evidence_cache import generate_evidence_cache
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
    name="ball_refiner.generate_evidence",
    fields=(
        BoundaryPathField("store", PathRole.DATA, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True),
        BoundaryPathField("checkpoint", PathRole.CHECKPOINT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
        BoundaryPathField("output", PathRole.CACHE, PathDirection.OUTPUT, PathKind.DIRECTORY),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New absolute versioned evidence cache")
    parser.add_argument("--sources", nargs="+", choices=SOURCES, required=True)
    parser.add_argument("--splits", nargs="+", choices=tuple(SPLIT_CODES), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-candidates", type=int, required=True)
    parser.add_argument("--nms-kernel", type=int, required=True)
    parser.add_argument("--patch-size", type=int, required=True)
    parser.add_argument("--subpixel-refine", action=argparse.BooleanOptionalAction, required=True)
    args = parser.parse_args()
    if not all(path.is_absolute() for path in (args.store, args.checkpoint, args.output)):
        parser.error("All paths must be absolute")
    roots = RuntimePathRoots(
        project_root=args.output.parent, data_root=args.store.parent, checkpoint_root=args.checkpoint.parent,
        cache_root=args.output.parent, artifact_root=args.output.parent,
        output_root=args.output.parent, external_asset_root=args.output.parent,
    )
    paths = PATH_BOUNDARY.validate(
        {"store": args.store, "checkpoint": args.checkpoint, "output": args.output},
        resolver=PathResolver(roots),
    )
    result = generate_evidence_cache(
        store_directory=paths.declared("store").path, checkpoint=paths.declared("checkpoint").path,
        output=paths.declared("output").path, splits=tuple(args.splits), sources=tuple(args.sources),
        device=args.device, subpixel_refine=args.subpixel_refine, stride=args.stride, batch_size=args.batch_size,
        candidates=BallCandidateConfig(
            max_candidates=args.max_candidates, nms_kernel=args.nms_kernel, patch_size=args.patch_size,
        ),
    )
    print(f"Published {result}")


if __name__ == "__main__":
    main()
