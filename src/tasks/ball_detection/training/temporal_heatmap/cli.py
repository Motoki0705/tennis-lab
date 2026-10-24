"""CLI for explicit architecture, phase and optimizer-update budgets."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.tasks.ball_detection.data.augmentation.video import AugmentationConfig
from src.tasks.ball_detection.training.runtime.arguments import (
    add_runtime_arguments,
    runtime_from_arguments,
)
from src.tasks.ball_detection.training.temporal_heatmap.configuration import (
    UpdateBudget,
    load_model_config,
)
from src.tasks.ball_detection.training.temporal_heatmap.runner import (
    train_temporal_heatmap,
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
    name="ball_detection.train_temporal_heatmap",
    fields=(
        BoundaryPathField("manifest", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE,
                          must_exist=True, allow_role_root=True),
        BoundaryPathField("evaluation_manifest", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE,
                          must_exist=True, allow_role_root=True),
        BoundaryPathField("model_config", PathRole.PROJECT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
        BoundaryPathField("output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY),
        BoundaryPathField("augmentation_config", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE,
                          must_exist=True, allow_role_root=True, required=False),
        BoundaryPathField("initial_artifact", PathRole.CHECKPOINT, PathDirection.INPUT, PathKind.FILE,
                          must_exist=True, required=False),
        BoundaryPathField("resume", PathRole.OUTPUT, PathDirection.INPUT, PathKind.FILE, must_exist=True, required=False),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Shared-pack heatmap pretraining or independent fine-tuning")
    parser.add_argument("--architecture", choices=("dpt", "hybrid"), required=True)
    parser.add_argument("--phase", choices=("pretrain", "finetune"), required=True)
    for name in ("manifest", "evaluation-manifest", "model-config", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("updates", "input-frames", "microbatch-windows", "warmup-updates", "evaluation-interval", "checkpoint-interval", "seed"):
        parser.add_argument(f"--{name}", type=int, required=True)
    for name in ("learning-rate", "minimum-lr-ratio", "weight-decay", "gradient-clip", "sigma-ratio", "gamma", "distance-px"):
        parser.add_argument(f"--{name}", type=float, required=True)
    parser.add_argument("--augmentation", choices=("none", "video"), required=True)
    parser.add_argument("--augmentation-config", type=Path)
    parser.add_argument("--initial-artifact", type=Path)
    parser.add_argument("--initial-artifact-sha256")
    parser.add_argument("--parent-update", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-update", type=int)
    add_runtime_arguments(parser)
    args = parser.parse_args()
    if (args.augmentation == "video") != (args.augmentation_config is not None):
        parser.error("Specify --augmentation-config exactly when --augmentation video")
    budget = UpdateBudget(**{key: getattr(args, key) for key in UpdateBudget.__dataclass_fields__})
    declared = {field.name: getattr(args, field.name) for field in PATH_BOUNDARY.fields
                if getattr(args, field.name) is not None}
    if not all(path.is_absolute() for path in declared.values()):
        parser.error("Training input and output paths must be absolute")
    model_root, output_root = args.model_config.parent.resolve(), args.output.parent.resolve()
    checkpoint_root = args.initial_artifact.parent.resolve() if args.initial_artifact is not None else output_root
    roots = RuntimePathRoots(project_root=model_root, data_root=model_root, artifact_root=model_root,
                             checkpoint_root=checkpoint_root, output_root=output_root,
                             cache_root=output_root, external_asset_root=model_root)
    paths = PATH_BOUNDARY.validate(declared, resolver=PathResolver(roots), independent_artifact_inputs=True)
    for name in declared:
        setattr(args, name, paths.declared(name).path)
    train_temporal_heatmap(architecture=args.architecture, phase=args.phase, manifest=args.manifest,
        evaluation_manifest=args.evaluation_manifest, model_config=load_model_config(args.model_config, args.architecture),
        output=args.output, budget=budget, runtime=runtime_from_arguments(args), device=torch.device(args.device),
        augmentation=AugmentationConfig.load(args.augmentation_config) if args.augmentation_config else None,
        distance_px=args.distance_px, initial_artifact=args.initial_artifact, initial_artifact_sha256=args.initial_artifact_sha256,
        parent_update=args.parent_update, resume=args.resume, stop_after_update=args.stop_after_update)


if __name__ == "__main__":
    main()
