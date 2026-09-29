"""Compare the configured player detector with explicit COCO weights on Meiji.

GPU, shared training queue only. Each variant executes the standard pipeline
through person_detection, including the calibrated court ROI. Predictions,
source hashes, model hashes and settings live in separate run-owned stores;
no input clip or previous observation run is modified. Tracking/pose do not
run. The #944 labels are incomplete: see PartialDetectionMetrics for what the
reported coverage counts mean. Failures are recorded and raised.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tasks.player_association.evaluation.dataset_labels import discover_labels
from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tasks.player_detection.evaluation.partial_labels import PartialDetectionMetrics
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.definition import standard_definition
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.tennis_scene.pipeline.runner import ComponentRunner
from src.tennis_scene.pipeline.source import build_clip_source
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.utils.checksum import dual_sha256


def compose_detector_runtime(
    repo: Path, report: Path, device: str, overrides: list[str], name: str,
) -> tuple[PipelineRuntimeConfig, list[str]]:
    code_root = Path(__file__).resolve().parents[2]
    applied = [
        f"paths.project_root={code_root}", f"paths.data_root={repo / 'data'}",
        f"paths.checkpoint_root={repo / 'ckpt'}", f"paths.external_asset_root={repo / 'third_party'}",
        f"paths.artifact_root={report}", f"paths.output_root={report}", f"paths.cache_root={report / 'cache'}",
        f"device={device}", "output_directory=run", "ball_detection.enabled=false", "execution.court_side=load",
        "player_reconstruction.enabled=false", "ball_reconstruction.enabled=false", "gvhmr.enabled=false", *overrides,
    ]
    with initialize_config_dir(version_base="1.3", config_dir=str(code_root / "src/tennis_scene/configs")):
        config = compose(config_name="pipeline", overrides=applied)
    (report / f"{name}.pipeline_config.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    return PipelineRuntimeConfig.from_config(config, bind_inputs=False), applied


def evaluate_clip(
    runtime: PipelineRuntimeConfig, source: ClipSource, labels: ClipLabels,
    store_root: Path, totals: PartialDetectionMetrics,
) -> dict[str, Any]:
    if source.clip_id != labels.clip_id or source.num_frames != labels.num_frames or set(source.camera_ids) != set(labels.cameras):
        raise ValueError("Source clip/camera/timeline differs from evaluation labels")
    nodes = standard_definition(runtime, source, code_identity=TennisSceneOrchestrator(runtime).code_identity)
    store = ClipStore(store_root, json_value(source))
    runner = ComponentRunner(nodes, store)
    runner.run(targets=tuple(f"person_detection/{camera}" for camera in source.camera_ids))
    reference = store.active("court_calibration")
    if reference is None:
        raise RuntimeError("No adopted court calibration")
    calibration = store.load(reference, ArtifactCodec(CourtCalibrationOutput))
    cameras: dict[str, Any] = {}
    for camera in source.camera_ids:
        reference = store.active(f"person_detection/{camera}")
        if reference is None:
            raise RuntimeError(f"No adopted detections for {camera}")
        detections = store.load(reference, ArtifactCodec(PersonDetectionOutput))
        if detections.camera_id != camera or len(detections.frame_offsets) != labels.num_frames + 1:
            raise ValueError(f"{camera} detection timeline differs from labels")
        metric = PartialDetectionMetrics(totals.min_iou, totals.score_threshold)
        for frame in range(labels.num_frames):
            start, end = detections.frame_offsets[frame:frame + 2]
            metric.update(detections.boxes_xyxy[start:end], detections.confidence[start:end],
                          labels=labels.cameras[camera], roles=labels.roles, frame=frame)
        totals.merge(metric)
        cameras[camera] = {
            "artifact": json_value(reference), "metrics": metric.compute(),
            "court_roi_available": calibration.footpoint_polygons[camera] is not None,
        }
    return {"status": "ok", "store": str(store_root), "seconds": runner.seconds, "cameras": cameras}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="Asset root holding data/, ckpt/, third_party/")
    parser.add_argument("--dataset", type=Path, required=True, help="Meiji structured dataset")
    parser.add_argument("--report", type=Path, required=True, help="New run-owned output directory")
    parser.add_argument("--clip", action="append", default=[], help="Subset of labelled clip IDs (repeatable)")
    parser.add_argument("--legacy-checkpoint", default="dino/checkpoint0029_4scale_swin.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-iou", type=float, default=.5)
    parser.add_argument("--override", action="append", default=[], help="Shared pipeline override")
    args = parser.parse_args()
    repo, dataset, report = args.repo.resolve(), args.dataset.resolve(), args.report.resolve()
    report.mkdir(parents=True, exist_ok=True)
    if (report / "comparison.json").exists():
        raise FileExistsError(f"Completed comparison already exists: {report}")
    paths = discover_labels(dataset)
    labelled = {path: ClipLabels.load(path) for path in paths}
    if args.clip:
        unknown = set(args.clip) - {labels.clip_id for labels in labelled.values()}
        if unknown:
            raise ValueError(f"Unknown labelled clips: {sorted(unknown)}")
        labelled = {path: labels for path, labels in labelled.items() if labels.clip_id in args.clip}
    if not labelled or len({labels.clip_id for labels in labelled.values()}) != len(labelled):
        raise ValueError("Expected nonempty, unique labelled clips")
    summary: dict[str, Any] = {
        "schema": "player_detection_old_box_agreement_v2", "status": "running",
        "min_iou": args.min_iou, "scope": "pipeline_court_roi",
        "labels": {labels.clip_id: {"path": str(path), "sha256": dual_sha256(path)}
                   for path, labels in labelled.items()}, "variants": {},
    }
    for name, extra in (
        ("coco_person", [f"people_models.dino_checkpoint={args.legacy_checkpoint}"]), ("player_ft", []),
    ):
        runtime, overrides = compose_detector_runtime(repo, report, args.device, [*args.override, *extra], name)
        if runtime.people.detector != "dino":
            raise ValueError("This comparison requires two DINO-format checkpoints")
        metric = PartialDetectionMetrics(args.min_iou, runtime.people.runtime.dino_detector.confidence)
        variant: dict[str, Any] = {"config_overrides": overrides, "clips": {}}
        summary["variants"][name] = variant
        for labels in labelled.values():
            video, clip = labels.clip_id.split("/")
            try:
                manifest = ClipManifest.load(dataset / "videos" / video / "clips" / clip)
                source = build_clip_source(tuple(manifest.media_path(camera) for camera in manifest.camera_ids),
                                           tuple(manifest.camera_ids), clip_id=manifest.clip_id)
                result = evaluate_clip(runtime, source, labels, report / name / "stores" / labels.clip_id, metric)
            except Exception as error:
                variant["clips"][labels.clip_id] = {
                    "status": "failed", "error_type": type(error).__name__, "error": str(error),
                }
                summary["status"] = "failed"
                write_json_atomic(report / "progress.json", summary)
                raise
            variant["clips"][labels.clip_id] = result
            variant["metrics"] = metric.compute()
            write_json_atomic(report / "progress.json", summary)
            print(json.dumps({"variant": name, "clip": labels.clip_id, "metrics": metric.compute()}), flush=True)
    summary["status"] = "ok"
    write_json_atomic(report / "comparison.json", summary)


if __name__ == "__main__":
    main()
