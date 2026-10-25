"""Evaluate one fixed heatmap artifact, calibrating only on validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.tasks.ball_detection.evaluation.temporal_heatmap import (
    evaluate_temporal_heatmap,
)
from src.tasks.ball_detection.model_io.mdd_heatmap import restore_heatmap_model
from src.tasks.ball_detection.training.runtime.arguments import (
    add_runtime_arguments,
    runtime_from_arguments,
)
from src.tasks.ball_detection.training.runtime.compilation import (
    compile_coordinate_model,
)
from src.tasks.ball_detection.training.runtime.images import jpeg_decoder_contract
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

PATH_BOUNDARY = NonHydraPathBoundary(
    name="ball_detection.evaluate_temporal_heatmap",
    fields=(
        BoundaryPathField("artifact", PathRole.CHECKPOINT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
        BoundaryPathField("manifest", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE,
                          must_exist=True, allow_role_root=True),
        BoundaryPathField("calibration", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE,
                          must_exist=True, allow_role_root=True, required=False),
        BoundaryPathField("output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.FILE),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a fixed temporal heatmap artifact using owner windows")
    for name in ("artifact", "manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--calibration", type=Path, help="Validation report for this exact artifact; required for test")
    parser.add_argument("--distance-px", type=float, required=True)
    add_runtime_arguments(parser)
    args = parser.parse_args()
    if args.split == "test" and args.calibration is None:
        parser.error("Test requires --calibration from validation; never calibrate on test")
    declared = {field.name: getattr(args, field.name) for field in PATH_BOUNDARY.fields
                if getattr(args, field.name) is not None}
    if not all(path.is_absolute() for path in declared.values()):
        parser.error("Evaluation input and output paths must be absolute")
    artifact_root, output_root = args.artifact.parent.resolve(), args.output.parent.resolve()
    roots = RuntimePathRoots(project_root=artifact_root, data_root=artifact_root, artifact_root=artifact_root,
                             checkpoint_root=artifact_root, output_root=output_root,
                             cache_root=output_root, external_asset_root=artifact_root)
    paths = PATH_BOUNDARY.validate(declared, resolver=PathResolver(roots), independent_artifact_inputs=True)
    for name in declared:
        setattr(args, name, paths.declared(name).path)
    if args.output.exists() or args.output.with_suffix(".frames.jsonl").exists():
        parser.error("Evaluation output already exists")
    runtime = runtime_from_arguments(args)
    device = torch.device(args.device)
    runtime.configure(device)
    saved = torch.load(args.artifact, map_location="cpu", weights_only=True)
    model = restore_heatmap_model(saved).to(device)
    training = saved.get("training")
    if (training is None or training["image_decode"] != jpeg_decoder_contract(runtime.jpeg_decoder)
            or training["runtime"]["precision"] != runtime.precision):
        raise ValueError("Evaluation must use the training artifact's saved JPEG decoder/version and precision")
    manifest = json.loads(args.manifest.read_text())
    if (manifest["ball_store"] != training["data"]["snapshot"]
            or {r["clip"]["clip_id"]: r["rgb_shard_sha256"] for r in manifest["clips"]} != training["data"]["rgb_shards_sha256"]
            or any(manifest["summaries"][split][key] != digest
                   for split, identities in training["data"]["summaries"].items() for key, digest in identities.items())):
        raise ValueError("Evaluation snapshot/common sets differ from the training artifact")
    compile_coordinate_model(model, mode=runtime.compile_mode, recompile_limit=runtime.compile_recompile_limit)
    calibration = json.loads(args.calibration.read_text()) if args.calibration else None
    evaluate_temporal_heatmap(model, args.manifest, split=args.split, runtime=runtime, device=device,
        artifact_sha256=dual_sha256(args.artifact), distance_px=args.distance_px,
        calibration_report=calibration, output=args.output)
