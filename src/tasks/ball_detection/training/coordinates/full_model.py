"""Explicit new-run initialization from a complete adopted query checkpoint."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

from src.tasks.ball_detection.checkpoints.initialization import initialize_full_model
from src.tasks.ball_detection.models.mdd import DeepMDDQueryDetector
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

PATH_BOUNDARY = NonHydraPathBoundary(name="ball_detection.full_model_mdd_query", fields=(
    BoundaryPathField("manifest", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
    BoundaryPathField("validation_manifest", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
    BoundaryPathField("initial_checkpoint", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
    BoundaryPathField("augmentation_config", PathRole.PROJECT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
    BoundaryPathField("output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY),
    BoundaryPathField("resume", PathRole.OUTPUT, PathDirection.INPUT, PathKind.FILE, must_exist=True, required=False),
))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-model initialization with fresh optimizer and fixed validation")
    names = ("manifest", "validation-manifest", "initial-checkpoint", "augmentation-config", "output")
    for name in names:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--initial-checkpoint-sha256", required=True)
    add_training_arguments(parser)
    args = parser.parse_args()
    validate_training_arguments(parser, args)
    if not re.fullmatch(r"[0-9a-f]{64}", args.initial_checkpoint_sha256):
        parser.error("Initial checkpoint SHA-256 must be explicitly pinned")
    for name in (*[n.replace("-", "_") for n in names], "resume"):
        path = getattr(args, name)
        if path is not None and not path.is_absolute():
            parser.error(f"{name} must be absolute")
    if args.resume is not None and args.resume.parent != args.output:
        parser.error("Resume is restricted to this run, not the adopted parent checkpoint")
    return args


def main() -> None:
    args = arguments()
    artifact_root = Path(os.path.commonpath((args.manifest.parent, args.validation_manifest.parent, args.initial_checkpoint.parent)))
    roots = RuntimePathRoots(project_root=args.augmentation_config.parent, data_root=args.manifest.parent,
        artifact_root=artifact_root, checkpoint_root=args.output.parent, cache_root=args.output.parent,
        output_root=args.output.parent, external_asset_root=args.output.parent)
    names = ("manifest", "validation_manifest", "initial_checkpoint", "augmentation_config", "output", "resume")
    paths = PATH_BOUNDARY.validate({name: getattr(args, name) for name in names if getattr(args, name) is not None}, resolver=PathResolver(roots))
    for name in names:
        if getattr(args, name) is not None:
            setattr(args, name, paths.declared(name).path)

    def initialize() -> tuple[DeepMDDQueryDetector, dict[str, Any]]:
        return initialize_full_model(args.initial_checkpoint, expected_sha256=args.initial_checkpoint_sha256, jpeg_decoder=args.jpeg_decoder)

    run_finetuning(args, validation_manifest=args.validation_manifest, initialize=initialize)
