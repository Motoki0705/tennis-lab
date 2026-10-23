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
    train_temporal_heatmap(architecture=args.architecture, phase=args.phase, manifest=args.manifest,
        evaluation_manifest=args.evaluation_manifest, model_config=load_model_config(args.model_config, args.architecture),
        output=args.output, budget=budget, runtime=runtime_from_arguments(args), device=torch.device(args.device),
        augmentation=AugmentationConfig.load(args.augmentation_config) if args.augmentation_config else None,
        distance_px=args.distance_px, initial_artifact=args.initial_artifact, initial_artifact_sha256=args.initial_artifact_sha256,
        parent_update=args.parent_update, resume=args.resume, stop_after_update=args.stop_after_update)


if __name__ == "__main__":
    main()
