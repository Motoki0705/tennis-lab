"""Evaluate a hybrid checkpoint under its saved decoder, labels and threshold."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from src.tasks.ball_detection.checkpoints.hybrid_training import restore_hybrid_model
from src.tasks.ball_detection.data.windows.dataset import FrozenBallWindowDataset
from src.tasks.ball_detection.evaluation.hybrid import evaluate_hybrid
from src.tasks.ball_detection.training.runtime.execution import CoordinateRuntime
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

PATH_BOUNDARY = NonHydraPathBoundary(name="ball_detection.evaluate_hybrid", fields=(
    BoundaryPathField("checkpoint", PathRole.CHECKPOINT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
    BoundaryPathField("manifest", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
    BoundaryPathField("output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.FILE),
))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if not all(p.is_absolute() for p in (args.checkpoint, args.manifest, args.output)):
        parser.error("Checkpoint, frozen hybrid manifest and output paths must be absolute")
    roots = RuntimePathRoots(project_root=args.checkpoint.parent, data_root=args.manifest.parent,
        checkpoint_root=args.checkpoint.parent, artifact_root=args.manifest.parent, output_root=args.output.parent,
        cache_root=args.output.parent, external_asset_root=args.output.parent)
    paths = PATH_BOUNDARY.validate({name: getattr(args, name) for name in ("checkpoint", "manifest", "output")},
                                   resolver=PathResolver(roots))
    for name in ("checkpoint", "manifest", "output"):
        setattr(args, name, paths.declared(name).path)
    if args.output.exists():
        raise ValueError("Evaluation output must be new")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model, recipe = restore_hybrid_model(saved)
    if dual_sha256(args.manifest) != recipe["hybrid_manifest_sha256"]:
        raise ValueError("Evaluation must use the saved frozen hybrid windows")
    runtime = replace(CoordinateRuntime(**recipe["runtime"]), num_workers=args.num_workers)
    device = torch.device(args.device)
    runtime.configure(device)
    model.to(device)
    data = FrozenBallWindowDataset(args.manifest, split=args.split, requires_pose=False, jpeg_decoder=runtime.jpeg_decoder)
    loader = runtime.loader(data, batch_size=recipe["batch_size"])
    report = evaluate_hybrid(model, loader, device, frame_steps=data.frame_steps,
        presence_threshold=recipe["presence_threshold"], loss_config=recipe["loss"],
        precision=runtime.precision, image_prefetch=runtime.image_prefetch)
    report.update(checkpoint_sha256=dual_sha256(args.checkpoint), manifest_sha256=dual_sha256(args.manifest),
                  split=args.split, input_contract=recipe["input_contract"], image_decode=recipe["image_decode"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
