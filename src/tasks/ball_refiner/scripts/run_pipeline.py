"""Run one video's detector -> full GMM component chain. CUDA requires the queue."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

from src.tasks.ball_refiner.deployment import CENTRE_SELECTION, load_inference_bundle
from src.tennis_scene.configuration import build_ball_detection_config
from src.tennis_scene.pipeline.artifacts import document_digest, json_value
from src.tennis_scene.pipeline.ball_refiner_recipe import ball_refiner_definition
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.pipeline.runner import ComponentRunner
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
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
from src.utils.paths import PROJECT_ROOT
from src.utils.video import probe_video_info

PATH_BOUNDARY = NonHydraPathBoundary(
    name="ball_refiner.run_pipeline",
    fields=(
        BoundaryPathField("video", PathRole.DATA, PathDirection.INPUT, PathKind.FILE, must_exist=True),
        BoundaryPathField("bundle", PathRole.CHECKPOINT, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True),
        BoundaryPathField("detector_checkpoint", PathRole.CHECKPOINT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
        BoundaryPathField("store", PathRole.ARTIFACT, PathDirection.OUTPUT, PathKind.DIRECTORY),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("video", "bundle", "detector-checkpoint", "store"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--source", choices=("execute", "load"), required=True)
    parser.add_argument("--detector-batch-size", type=int, required=True)
    parser.add_argument("--refiner-batch-size", type=int, required=True)
    args = parser.parse_args()
    if not all(getattr(args, name).is_absolute() for name in ("video", "bundle", "detector_checkpoint", "store")):
        parser.error("All paths must be absolute")
    checkpoint_root = Path(os.path.commonpath([args.bundle.parent, args.detector_checkpoint.parent]))
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT, data_root=args.video.parent, checkpoint_root=checkpoint_root,
        cache_root=args.store.parent, artifact_root=args.store.parent,
        output_root=args.store.parent, external_asset_root=PROJECT_ROOT / "third_party",
    )
    resolver = PathResolver(roots)
    paths = PATH_BOUNDARY.validate({name: getattr(args, name) for name in (
        "video", "bundle", "detector_checkpoint", "store",
    )}, resolver=resolver)
    video_path = paths.declared("video").path
    bundle_path = paths.declared("bundle").path
    store_path = paths.declared("store").path
    detector_path = paths.declared("detector_checkpoint").path
    if args.source == "load" and not (store_path / "scene.json").is_file():
        raise FileNotFoundError("load requires an existing component store")
    bundle = load_inference_bundle(bundle_path)
    required = bundle.detector
    # The scene YAML owns incidental detector options (legacy point gate, prefetch).
    # Every refiner-critical field is explicitly fixed by the exported contract.
    config = cast(dict[str, Any], OmegaConf.to_container(
        OmegaConf.load(PROJECT_ROOT / "src/tennis_scene/configs/pipeline.yaml")["ball_detection"], resolve=True,
    ))
    config.update({
        "checkpoint": str(detector_path.relative_to(checkpoint_root)), "batch_size": args.detector_batch_size,
        "image_size": list(required.image_size_hw), "normalize_imagenet": required.normalization.enabled,
        "subpixel_refine": required.subpixel_refine, "checkpoint_strict": True,
        "window_stride": required.stride, "tail_policy": "backfill", "overlap_aggregation": CENTRE_SELECTION,
        "candidates": asdict(required.candidates),
    })
    detector = build_ball_detection_config(config, resolver, device=args.device)
    info = probe_video_info(video_path)
    source = ClipSource(video_path.stem, (SourceVideo(
        args.camera_id, video_path, dual_sha256(video_path), info.frame_count, info.fps, info.width, info.height,
    ),))
    source_code = PROJECT_ROOT / "src"
    code_identity = document_digest({
        str(path.relative_to(source_code)): dual_sha256(path) for path in sorted(source_code.rglob("*.py"))
    })
    nodes = ball_refiner_definition(
        source, detector_config=detector, bundle_directory=bundle_path, batch_size=args.refiner_batch_size,
        code_identity=code_identity, execution_source=args.source,
    )
    store = ClipStore(store_path, json_value(source))
    runner = ComponentRunner(nodes, store)
    references = runner.run()
    print(json.dumps({"status": "complete", "scene_index": str(store.index_path), "source": json_value(source),
                      "components": runner.statuses, "seconds": runner.seconds, "references": json_value(references)}))


if __name__ == "__main__":
    main()
