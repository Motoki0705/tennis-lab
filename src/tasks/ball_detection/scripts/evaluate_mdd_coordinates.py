"""Evaluate a frozen coordinate checkpoint on val/test with full/common FPS reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.tasks.ball_detection.data.coordinate_dataset import (
    CoordinateWindowDataset,
    collate_coordinate_windows,
)
from src.tasks.ball_detection.models.mdd_pose import (
    MDDPoseConfig,
    MDDPoseDetector,
    MDDQueryDetector,
)
from src.tasks.ball_detection.training.coordinate_evaluation import evaluate_coordinates
from src.tasks.base.visualization.inference_queue import shared_repository_root
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
    name="ball_detection.evaluate_mdd_coordinates",
    fields=(
        BoundaryPathField("checkpoint", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
        BoundaryPathField("manifest", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
        BoundaryPathField("output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.FILE),
        BoundaryPathField("artifact_root", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.DIRECTORY,
                          must_exist=True, allow_role_root=True),
        BoundaryPathField("output_root", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY,
                          allow_role_root=True),
    ),
)


def main() -> None:
    project = shared_repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=project / "outputs")
    parser.add_argument("--output-root", type=Path, default=project / "outputs")
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    if not all(path.is_absolute() for path in (args.checkpoint, args.manifest, args.output, args.artifact_root, args.output_root)):
        parser.error("Artifact and output paths must be absolute")
    roots = RuntimePathRoots(project_root=project, data_root=project / "data", artifact_root=args.artifact_root,
                             output_root=args.output_root, checkpoint_root=project / "ckpt", cache_root=project / ".cache",
                             external_asset_root=project / "third_party")
    paths = PATH_BOUNDARY.validate({key: getattr(args, key) for key in ("checkpoint", "manifest", "output", "artifact_root", "output_root")},
                                   resolver=PathResolver(roots))
    checkpoint, manifest, output = (paths.declared(key).path for key in ("checkpoint", "manifest", "output"))
    if output.exists():
        raise ValueError("Evaluation output already exists; choose a new report path")
    checkpoint_hash = dual_sha256(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("schema") != "mdd_coordinates.v2":
        raise ValueError("Expected a v2 coordinate checkpoint with declared FPS/MDD training provenance")
    manifest_hash = dual_sha256(manifest)
    if manifest_hash != saved["manifest_sha256"]:
        raise ValueError("Evaluation manifest differs from the checkpoint's frozen data/split identity")
    config = MDDPoseConfig(**saved["model_config"])
    model = MDDPoseDetector(config) if config.requires_pose else MDDQueryDetector(config)
    model.load_state_dict(saved["state_dict"], strict=True)
    device = torch.device(args.device)
    model.to(device)
    dataset = CoordinateWindowDataset(manifest, split=args.split, requires_pose=config.requires_pose,
                                      mdd_a=saved["mdd_a"], mdd_b=saved["mdd_b"])
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_coordinate_windows, num_workers=0)
    report = evaluate_coordinates(model, loader, device, dataset.frame_steps)
    if dual_sha256(checkpoint) != checkpoint_hash or dual_sha256(manifest) != manifest_hash:
        raise ValueError("Checkpoint or manifest changed during evaluation")
    report.update(checkpoint_sha256=checkpoint_hash, manifest_sha256=manifest_hash, split=args.split)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(output), split=args.split, scopes=report["scopes"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
