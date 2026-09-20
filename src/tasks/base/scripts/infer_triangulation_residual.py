"""Infer geometric residuals and compare a real clip with its triangulated seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tasks.base.triangulation_residual.inference import evaluate_clip
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
    name="base.triangulation_residual.inference",
    fields=(
        BoundaryPathField(
            "source",
            PathRole.CHECKPOINT,
            PathDirection.INPUT,
            PathKind.ANY,
            must_exist=True,
        ),
        BoundaryPathField(
            "clip",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
        ),
        BoundaryPathField(
            "output", PathRole.ARTIFACT, PathDirection.OUTPUT, PathKind.DIRECTORY
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("plcs", "blcs"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--run-dir", type=Path)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    source_path = args.checkpoint if args.checkpoint is not None else args.run_dir
    if not all(path.is_absolute() for path in (source_path, args.clip, args.output)):
        raise ValueError("Source, clip and output must be explicit absolute paths")
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT,
        data_root=args.clip.parent.resolve(),
        checkpoint_root=source_path.parent.resolve(),
        artifact_root=args.output.parent.resolve(),
        output_root=args.output.parent.resolve(),
        cache_root=PROJECT_ROOT,
        external_asset_root=PROJECT_ROOT,
    )
    validated = PATH_BOUNDARY.validate(
        {"source": source_path, "clip": args.clip, "output": args.output},
        resolver=PathResolver(roots),
    )
    source_path = validated.declared("source").path
    checkpoint = (
        source_path
        if args.checkpoint is not None
        else Path(
            json.loads((source_path / "evaluation.json").read_text())["checkpoint"]
        )
    )
    if args.run_dir is not None and not checkpoint.resolve().is_relative_to(
        source_path
    ):
        raise ValueError(
            "Selected best checkpoint is outside the declared run directory"
        )
    evaluate_clip(
        args.task,
        checkpoint,
        validated.declared("clip").path,
        validated.declared("output").path,
        device=args.device,
        render=not args.no_render,
    )


if __name__ == "__main__":
    main()
