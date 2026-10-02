"""Validation-only checkpoint selection. CUDA execution must use the training queue."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import torch

from src.tasks.ball_refiner.evaluation.detector_selection import (
    CHECKPOINT_NAMES,
    SelectionCheckpoint,
    compare_detectors,
    prepare_comparison,
)
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
    name="ball_refiner.compare_detectors",
    fields=(
        BoundaryPathField("store", PathRole.DATA, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True),
        BoundaryPathField("cache_manifest", PathRole.CACHE, PathDirection.INPUT, PathKind.FILE, must_exist=True),
        BoundaryPathField("ft_e13", PathRole.CHECKPOINT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
        BoundaryPathField("mixed_e0", PathRole.CHECKPOINT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
        BoundaryPathField("mixed_e11", PathRole.CHECKPOINT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
        BoundaryPathField("output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    for name in CHECKPOINT_NAMES:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-sha256", nargs=3, required=True, metavar="SHA256", help="ft-e13, mixed-e0, mixed-e11, in this order")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, required=True)
    parser.add_argument("--cpu-threads", type=int, choices=range(1, 5), required=True)
    parser.add_argument("--dry-run", action="store_true", help="CPU identity/split validation; no model or output creation")
    args = parser.parse_args()
    names = ("store", "cache_manifest", *(name.replace("-", "_") for name in CHECKPOINT_NAMES), "output")
    values = {name: getattr(args, name) for name in names}
    if not all(path.is_absolute() for path in values.values()):
        parser.error("All paths must be absolute")
    checkpoint_root = Path(os.path.commonpath([values[name.replace("-", "_")].parent for name in CHECKPOINT_NAMES]))
    roots = RuntimePathRoots(
        project_root=args.output.parent, data_root=args.store.parent, checkpoint_root=checkpoint_root,
        cache_root=args.cache_manifest.parent, artifact_root=args.output.parent,
        output_root=args.output.parent, external_asset_root=args.output.parent,
    )
    paths = PATH_BOUNDARY.validate(values, resolver=PathResolver(roots))
    checkpoints = tuple(SelectionCheckpoint(name, paths.declared(name.replace("-", "_")).path, sha)
                        for name, sha in zip(CHECKPOINT_NAMES, args.expected_sha256, strict=True))
    store, cache = paths.declared("store").path, paths.declared("cache_manifest").path
    torch.set_num_threads(args.cpu_threads)
    cv2.setNumThreads(args.cpu_threads)
    if args.dry_run:
        _, _, manifest = prepare_comparison(store, cache, checkpoints)
        print(json.dumps(manifest, indent=2))
        return
    result = compare_detectors(
        store_directory=store, cache_manifest=cache, checkpoints=checkpoints, output=paths.declared("output").path,
        device=args.device, batch_size=args.batch_size, cuda_allocator_limit_gib=args.cuda_allocator_limit_gib,
    )
    print(f"Published {result}")


if __name__ == "__main__":
    main()
