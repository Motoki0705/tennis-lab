from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from src.tasks.ball_detection.checkpoints.pretraining import (
    completed_pretraining,
    pretraining_config,
    transfer_encoder,
)
from src.tasks.ball_detection.models.mdd import DeepMDDQueryDetector
from src.tasks.ball_detection.preprocessing import RGBToMDD
from src.tasks.ball_detection.training.runtime.images import (
    jpeg_decoder_contract,
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

from .joint import add_training_arguments, run_finetuning, validate_training_arguments

PATH_BOUNDARY = NonHydraPathBoundary(name="ball_detection.posttrain_mdd_query", fields=(
    BoundaryPathField("manifest", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
    BoundaryPathField("pretraining_run", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True),
    BoundaryPathField("augmentation_config", PathRole.PROJECT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
    BoundaryPathField("output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY),
    BoundaryPathField("resume", PathRole.OUTPUT, PathDirection.INPUT, PathKind.FILE, must_exist=True, required=False),
))


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Frozen CNN warmup then joint fine-tuning; augmentation only in train")
    for key in ("manifest", "pretraining-run", "augmentation-config", "output"):
        p.add_argument(f"--{key}", type=Path, required=True)
    add_training_arguments(p)
    args = p.parse_args()
    validate_training_arguments(p, args)
    for name in ("manifest", "pretraining_run", "augmentation_config", "output", "resume"):
        value = getattr(args, name)
        if value is not None and not value.is_absolute():
            p.error(f"{name} must be absolute")
    return args




def main() -> None:
    args = arguments()
    roots = RuntimePathRoots(project_root=args.augmentation_config.parent, data_root=args.manifest.parent,
        artifact_root=Path(os.path.commonpath((args.manifest.parent, args.pretraining_run.parent))),
        checkpoint_root=args.output.parent, cache_root=args.output.parent,
        output_root=args.output.parent, external_asset_root=args.output.parent)
    paths = PATH_BOUNDARY.validate({k: getattr(args, k) for k in ("manifest", "pretraining_run", "augmentation_config", "output", "resume")
                                   if getattr(args, k) is not None}, resolver=PathResolver(roots))
    for key in ("manifest", "pretraining_run", "augmentation_config", "output", "resume"):
        if getattr(args, key) is not None:
            setattr(args, key, paths.declared(key).path)
    parent, pretrained = completed_pretraining(args.pretraining_run, args.manifest)
    config = pretraining_config(pretrained)
    if pretrained["recipe"]["image_decode"] != jpeg_decoder_contract(args.jpeg_decoder):
        raise ValueError("Posttraining JPEG decoder must match pretraining exactly")
    def initialize() -> tuple[DeepMDDQueryDetector, dict[str, Any]]:
        model = DeepMDDQueryDetector(config)
        model.mdd = RGBToMDD.from_contract(pretrained["recipe"]["input_contract"])
        transfer = transfer_encoder(parent, model)
        return model, transfer

    run_finetuning(args, validation_manifest=args.manifest, initialize=initialize)
