"""Evaluate a frozen coordinate checkpoint on val/test with full/common FPS reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.tasks.ball_detection.data.coordinate_dataset import (
    CoordinateWindowDataset,
)
from src.tasks.ball_detection.models.mdd_pose import (
    MDDPoseConfig,
    MDDPoseDetector,
    MDDQueryDetector,
)
from src.tasks.ball_detection.training.coordinate_checkpoint import (
    validate_coordinate_checkpoint,
)
from src.tasks.ball_detection.training.coordinate_compilation import (
    COMPILE_MODES,
    coordinate_compilation_report,
)
from src.tasks.ball_detection.training.coordinate_evaluation import evaluate_coordinates
from src.tasks.ball_detection.training.coordinate_images import (
    JPEG_DECODERS,
    jpeg_decoder_contract,
)
from src.tasks.ball_detection.training.coordinate_runtime import CoordinateRuntime
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
    parser.add_argument("--precision", choices=("fp32", "bf16"),
                        help="Defaults to the saved training precision")
    parser.add_argument("--compile-mode", choices=COMPILE_MODES,
                        help="Defaults to the saved compilation mode; off explicitly evaluates eagerly")
    parser.add_argument("--compile-recompile-limit", type=int)
    parser.add_argument("--jpeg-decoder", choices=JPEG_DECODERS, help="Defaults to the saved training JPEG decoder")
    parser.add_argument("--input-verification", choices=("lazy", "upfront"), default="upfront")
    parser.add_argument("--image-prefetch", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=2)
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
    preprocessing = validate_coordinate_checkpoint(saved)
    manifest_hash = dual_sha256(manifest)
    if manifest_hash != saved["manifest_sha256"]:
        raise ValueError("Evaluation manifest differs from the checkpoint's frozen data/split identity")
    config = MDDPoseConfig(**saved["model_config"])
    model = (MDDPoseDetector(config, mdd_a=preprocessing.a, mdd_b=preprocessing.b) if config.requires_pose
             else MDDQueryDetector(config, mdd_a=preprocessing.a, mdd_b=preprocessing.b))
    model.load_state_dict(saved["state_dict"], strict=True)
    device = torch.device(args.device)
    saved_runtime = saved["training_state"]["recipe"]["runtime"]
    saved_precision = saved_runtime["precision"]
    precision = saved_precision if args.precision is None else args.precision
    runtime = CoordinateRuntime(precision=precision, num_workers=args.num_workers, pin_memory=args.pin_memory,
                                prefetch_factor=args.prefetch_factor, cpu_threads=args.cpu_threads,
                                compile_mode=saved_runtime["compile_mode"] if args.compile_mode is None else args.compile_mode,
                                compile_recompile_limit=saved_runtime["compile_recompile_limit"]
                                if args.compile_recompile_limit is None else args.compile_recompile_limit,
                                jpeg_decoder=saved_runtime["jpeg_decoder"] if args.jpeg_decoder is None else args.jpeg_decoder,
                                input_verification=args.input_verification,
                                image_prefetch=saved_runtime["image_prefetch"] if args.image_prefetch is None else args.image_prefetch)
    dataset = CoordinateWindowDataset(manifest, split=args.split, requires_pose=config.requires_pose, jpeg_decoder=runtime.jpeg_decoder)
    loader = runtime.loader(dataset, batch_size=args.batch_size)
    runtime.configure(device)
    active_decoder = jpeg_decoder_contract(runtime.jpeg_decoder)
    if args.jpeg_decoder is None and active_decoder != saved["image_decode"]:
        raise ValueError("JPEG decoder implementation/version changed; use an explicit --jpeg-decoder override")
    model.to(device)
    runtime.configure_model(model)
    report = evaluate_coordinates(model, loader, device, dataset.frame_steps, precision=precision,
                                  image_prefetch=runtime.image_prefetch)
    if dual_sha256(checkpoint) != checkpoint_hash or dual_sha256(manifest) != manifest_hash:
        raise ValueError("Checkpoint or manifest changed during evaluation")
    report.update(checkpoint_sha256=checkpoint_hash, manifest_sha256=manifest_hash, split=args.split,
                  precision=precision, batch_size=args.batch_size, num_workers=args.num_workers,
                  image_decode=active_decoder, training_image_decode=saved["image_decode"],
                  input_contract=model.mdd.input_contract(), compilation=coordinate_compilation_report(model))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(output), split=args.split, scopes=report["scopes"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
