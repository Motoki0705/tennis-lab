"""Prepare annotated Meiji SceneResults or run the frozen-base motion ablation."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import torch

from src.tasks.ball_refiner_3d.post_training.dataset import (
    MeijiBenchmark,
    PreparedDataset,
    prepare_benchmark,
    sha256_file,
)
from src.tasks.ball_refiner_3d.post_training.fusion import FUSION_MODES
from src.tasks.ball_refiner_3d.post_training.runner import (
    PostTrainingConfig,
    run_post_training,
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
from src.utils.device import resolve_device
from src.utils.paths import PROJECT_ROOT

PATH_BOUNDARY = NonHydraPathBoundary(
    name="ball_refiner_3d.post_train",
    fields=(
        BoundaryPathField(
            "dataset",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "smplx",
            PathRole.CHECKPOINT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
            required=False,
        ),
        BoundaryPathField(
            "checkpoint",
            PathRole.CHECKPOINT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
            required=False,
        ),
        BoundaryPathField(
            "reference_labels",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
            required=False,
        ),
        BoundaryPathField(
            "reference_pins",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
            allow_role_root=True,
            required=False,
        ),
        BoundaryPathField(
            "output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY
        ),
    ),
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Verify/reduce v3 archives on CPU")
    run = sub.add_parser(
        "run", help="Fit one fusion configuration or evaluate the ball-only reference"
    )
    for command in (prepare, run):
        command.add_argument("--dataset", type=Path, required=True)
        command.add_argument("--manifest-sha256", required=True)
        command.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--smplx", type=Path, required=True)
    prepare.add_argument("--reference-labels", type=Path, required=True)
    prepare.add_argument("--reference-pins", type=Path, required=True)
    prepare.add_argument("--reference-pins-sha256", required=True)
    run.add_argument("--prepared-sha256", required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--checkpoint-sha256", required=True)
    run.add_argument("--mode", choices=(*FUSION_MODES, "ball-only"), required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--epochs", type=int, default=12)
    run.add_argument("--learning-rate", type=float, default=3e-4)
    run.add_argument("--device", choices=("cuda", "cpu"), required=True)
    args = parser.parse_args(argv)

    values = {
        name: value for name, value in vars(args).items() if isinstance(value, Path)
    }
    weights = args.smplx if args.command == "prepare" else args.checkpoint
    if not all(path.is_absolute() for path in values.values()):
        parser.error("All file and directory paths must be absolute")
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT,
        data_root=args.dataset,
        checkpoint_root=weights.parent,
        output_root=args.output.parent,
        artifact_root=PROJECT_ROOT,
        cache_root=PROJECT_ROOT,
        external_asset_root=PROJECT_ROOT,
    )
    paths = PATH_BOUNDARY.validate(
        values,
        resolver=PathResolver(roots),
        independent_artifact_inputs=True,
    )
    dataset_path = paths.declared("dataset").path
    output = paths.declared("output").path
    if output.is_relative_to(dataset_path):
        parser.error("The output must be separate from the input dataset")
    torch.set_num_threads(2)
    if args.command == "prepare":
        prepared = prepare_benchmark(
            MeijiBenchmark.load(dataset_path, args.manifest_sha256),
            paths.declared("smplx").path,
            paths.declared("reference_labels").path,
            paths.declared("reference_pins").path,
            args.reference_pins_sha256,
            output,
        )
        print(f"{sha256_file(prepared)}  {prepared}")
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    run_post_training(
        PreparedDataset.load(dataset_path, args.prepared_sha256, args.manifest_sha256),
        paths.declared("checkpoint").path,
        args.checkpoint_sha256,
        output,
        PostTrainingConfig(args.mode, args.seed, args.epochs, args.learning_rate),
        resolve_device(args.device),
        commit,
    )
    print(f"Completed {output}")


if __name__ == "__main__":
    main()
