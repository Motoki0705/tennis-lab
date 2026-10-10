"""Evaluate a Court checkpoint on synthetic test and real validation separately."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tasks.court_detection.ablation.evaluation import evaluate_checkpoint
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
    name="court_detection.ablation_evaluation",
    fields=(
        BoundaryPathField(
            "checkpoint",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY
        ),
        BoundaryPathField(
            "data_root",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
            required=False,
        ),
        BoundaryPathField(
            "checkpoint_root",
            PathRole.CHECKPOINT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
            required=False,
        ),
        BoundaryPathField(
            "external_asset_root",
            PathRole.EXTERNAL_ASSET,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
            required=False,
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    for name in ("data-root", "checkpoint-root", "external-asset-root"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    overrides = {
        name: getattr(args, name).expanduser().resolve()
        for name in ("data_root", "checkpoint_root", "external_asset_root")
        if getattr(args, name) is not None
    }
    output = args.output.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    binding_roots = {
        "data_root": PROJECT_ROOT,
        "checkpoint_root": PROJECT_ROOT,
        "external_asset_root": PROJECT_ROOT,
        **overrides,
    }
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT,
        data_root=binding_roots["data_root"],
        checkpoint_root=binding_roots["checkpoint_root"],
        external_asset_root=binding_roots["external_asset_root"],
        artifact_root=PROJECT_ROOT,
        output_root=output.parent,
        cache_root=PROJECT_ROOT,
    )
    paths = PATH_BOUNDARY.validate(
        {"checkpoint": checkpoint, "output": output, **overrides},
        resolver=PathResolver(roots),
        independent_artifact_inputs=True,
    )
    output = paths.declared("output").path
    evaluate_checkpoint(
        paths.declared("checkpoint").path,
        output,
        device=args.device,
        path_overrides={name: str(paths.declared(name).path) for name in overrides},
    )
    print(output / "evaluation.json")


if __name__ == "__main__":
    main()
