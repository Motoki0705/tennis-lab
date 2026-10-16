"""Refine an offline coordinate+mask NPZ using an explicit trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.tasks.ball_refiner_3d.inference.files import predict_file
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
    name="ball_refiner_3d.predict",
    fields=(
        BoundaryPathField(
            "checkpoint",
            PathRole.CHECKPOINT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
        ),
        BoundaryPathField(
            "input", PathRole.DATA, PathDirection.INPUT, PathKind.FILE, must_exist=True
        ),
        BoundaryPathField(
            "output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.FILE
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "input", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if not all(
        path.is_absolute() for path in (args.checkpoint, args.input, args.output)
    ):
        parser.error("checkpoint, input and output must be explicit absolute paths")
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT,
        data_root=args.input.parent,
        checkpoint_root=args.checkpoint.parent,
        output_root=args.output.parent,
        artifact_root=args.output.parent,
        cache_root=args.output.parent,
        external_asset_root=args.output.parent,
    )
    paths = PATH_BOUNDARY.validate(
        {"checkpoint": args.checkpoint, "input": args.input, "output": args.output},
        resolver=PathResolver(roots),
    )
    torch.set_num_threads(2)
    predict_file(
        paths.declared("checkpoint").path,
        paths.declared("input").path,
        paths.declared("output").path,
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    print(str(args.output))


if __name__ == "__main__":
    main()
