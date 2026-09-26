"""Court side on every clip of a structured dataset: detector ball vs. reviewed-annotation ball.

``observe`` (GPU, through the shared training queue) runs only the court
detection/calibration and ball detection nodes of the default ``pipeline.yaml``
into a run-owned store per clip. People and body features are disabled; they
are not side evidence. ``decide`` (CPU) reads those artifacts and applies the
``court_side`` component twice per clip: to the detector ball (the production
input) and to the reviewed ``outsource/<camera>_annotations.json`` ball
(``observed`` points only), which serves as the reference. Every decision or
explicit stop, with all hypothesis scores, goes to ``<report>/decisions.json``.
``decide`` can be repeated with ``--override court_side.<field>=<value>``; it
does not re-run any model. Nothing is written outside ``--report``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.generate_dataset.manifest import (
    ClipManifest,
    load_dataset_manifest,
)
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionOutput
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.court_side import CourtSideModule
from src.tennis_scene.pipeline.contracts import AssemblyContext, ClipSource
from src.tennis_scene.pipeline.definition import standard_definition
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.imports.ball_annotations import convert_ball_annotation
from src.tennis_scene.pipeline.input_assembly.reconstruction import (
    CourtSideInputAssembler,
)
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.tennis_scene.pipeline.runner import ComponentRunner
from src.tennis_scene.pipeline.source import build_clip_source
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec

CONFIG_DIR = Path(__file__).resolve().parents[2] / "src/tennis_scene/configs"
# Human-confirmed sides (knowledge run-scene-component-meiji-target2-20260925/side_confirmation.json).
CONFIRMED = {"video_000/clip_000": [False, False, True]}


def compose_runtime(repo: Path, report: Path, device: str, overrides: list[str]) -> tuple[PipelineRuntimeConfig, list[str]]:
    applied = [f"paths.project_root={CONFIG_DIR.parents[2]}", f"paths.data_root={repo / 'data'}",
        f"paths.checkpoint_root={repo / 'ckpt'}", f"paths.external_asset_root={repo / 'third_party'}",
        f"paths.artifact_root={report}", f"paths.output_root={report}", f"paths.cache_root={report / 'cache'}",
        f"device={device}", "output_directory=run", "person_observations.enabled=false",
        "player_reconstruction.enabled=false", "gvhmr.enabled=false", *overrides]
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        config = compose(config_name="pipeline", overrides=applied)
    (report / "pipeline_config.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    return PipelineRuntimeConfig.from_config(config, bind_inputs=False), applied


def clip_source(clip: Path) -> tuple[ClipManifest, ClipSource]:
    manifest = ClipManifest.load(clip)
    videos = tuple(manifest.media_path(camera) for camera in manifest.camera_ids)
    return manifest, build_clip_source(videos, tuple(manifest.camera_ids), clip_id=manifest.clip_id)


def observe(runtime: PipelineRuntimeConfig, code_identity: str, clip: Path, store_root: Path) -> dict[str, Any]:
    _, source = clip_source(clip)
    nodes = standard_definition(runtime, source, code_identity=code_identity)
    runner = ComponentRunner(nodes, ClipStore(store_root, json_value(source)))
    runner.run(targets=("court_calibration", *(f"ball_detection/{c}" for c in source.camera_ids)))
    return {"statuses": runner.statuses, "seconds": runner.seconds}


def side_record(module: CourtSideModule, assembler: CourtSideInputAssembler, source: ClipSource,
                calibration: CourtCalibrationOutput, balls: dict[str, BallDetectionOutput]) -> dict[str, Any]:
    visible = {camera: int(ball.observed.sum()) for camera, ball in balls.items()}
    inputs = assembler.assemble(AssemblyContext(source), {"calibration": calibration, **{f"ball_{c}": b for c, b in balls.items()}})
    try:
        side = module.process(inputs)
    except ReconstructionUnavailable as stopped:
        return {"decided": False, "reason": stopped.reason, "message": str(stopped), "diagnostics": json_value(stopped.diagnostics),
                "observed_frames": visible}
    return {"decided": True, "view_half_turns": list(side.view_half_turns), "margin": side.margin, "frames": side.frames,
            "hypotheses": [json_value(h) for h in side.hypotheses], "observed_frames": visible}


def decide(runtime: PipelineRuntimeConfig, clip: Path, store_root: Path) -> dict[str, Any]:
    manifest, source = clip_source(clip)
    store = ClipStore(store_root, json_value(source))
    reference = store.active("court_calibration")
    if reference is None:
        raise FileNotFoundError(f"{manifest.clip_id}: no court_calibration artifact; run observe first")
    calibration = store.load(reference, ArtifactCodec(CourtCalibrationOutput))
    module = CourtSideModule(source.camera_ids, runtime.court_side, max_frames=runtime.sampling_max_frames)
    assembler = CourtSideInputAssembler(runtime.ball_detection.score_threshold)
    detector: dict[str, BallDetectionOutput] = {}
    for camera in source.camera_ids:
        ball = store.active(f"ball_detection/{camera}")
        if ball is None:
            raise FileNotFoundError(f"{manifest.clip_id}: no ball_detection/{camera} artifact; run observe first")
        detector[camera] = store.load(ball, ArtifactCodec(BallDetectionOutput))
    annotated = {video.camera_id: convert_ball_annotation(clip / "outsource" / f"{video.camera_id}_annotations.json", video)[0]
                 for video in source.videos}
    record = {"clip_id": manifest.clip_id, "frames": source.num_frames, "camera_ids": list(calibration.calibration.camera_ids),
              "reference_camera": calibration.reference_camera, "excluded_cameras": calibration.calibration.excluded,
              "detector": side_record(module, assembler, source, calibration, detector),
              "annotation": side_record(module, assembler, source, calibration, annotated)}
    detected, annotation = record["detector"], record["annotation"]
    record["agreement"] = (detected["view_half_turns"] == annotation["view_half_turns"]
                           if detected["decided"] and annotation["decided"] else None)
    if manifest.clip_id in CONFIRMED:
        record["confirmed_half_turns"] = CONFIRMED[manifest.clip_id]
    return record


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"clips": len(records)}
    for name in ("detector", "annotation"):
        rows = [r[name] for r in records if name in r]
        reasons: dict[str, int] = {}
        for row in rows:
            if not row["decided"]:
                reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
        summary[name] = {"decided": sum(r["decided"] for r in rows), "stopped": reasons}
    compared = [r for r in records if r.get("agreement") is not None]
    summary["both_decided"] = len(compared)
    summary["agreements"] = sum(bool(r["agreement"]) for r in compared)
    summary["disagreeing_clips"] = [r["clip_id"] for r in compared if not r["agreement"]]
    summary["confirmed"] = {r["clip_id"]: {name: r[name].get("view_half_turns") == r["confirmed_half_turns"] for name in ("detector", "annotation")}
                            for r in records if "confirmed_half_turns" in r}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="Root holding data/, ckpt/ and third_party/")
    parser.add_argument("--dataset", type=Path, required=True, help="Structured dataset directory (dataset.json)")
    parser.add_argument("--report", type=Path, required=True, help="Run-owned directory: court_side/evaluate/<experiment>/<run-id>")
    parser.add_argument("--phase", choices=("observe", "decide", "both"), default="both")
    parser.add_argument("--clip", action="append", default=[], help="Restrict to clip IDs (repeatable)")
    parser.add_argument("--override", action="append", default=[], help="Extra pipeline.yaml override (decide thresholds)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    repo, dataset, report = args.repo.resolve(), args.dataset.resolve(), args.report.resolve()
    report.mkdir(parents=True, exist_ok=True)
    runtime, overrides = compose_runtime(repo, report, args.device, args.override)
    manifest = load_dataset_manifest(dataset)
    records = [manifest.clips[key] for key in sorted(manifest.clips) if not args.clip or key in args.clip]
    if args.clip and len(records) != len(args.clip):
        raise ValueError(f"Unknown clip IDs: {sorted(set(args.clip) - {r.clip_id for r in records})}")
    if args.phase in ("observe", "both"):
        code_identity = TennisSceneOrchestrator(runtime).code_identity
        observed: dict[str, Any] = {"schema": "court_side_observe_v1", "config_overrides": overrides, "code_sha256": code_identity, "clips": {}}
        for record in records:
            try:
                observed["clips"][record.clip_id] = {"status": "ok", **observe(runtime, code_identity, dataset / record.path, report / "stores" / record.clip_id)}
            except (ReconstructionUnavailable, ValueError) as error:
                # A clip whose court cannot be calibrated is reported, never skipped silently.
                observed["clips"][record.clip_id] = {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                                                     "reason": getattr(error, "reason", None)}
            write_json_atomic(report / "observe.json", observed)
            print(json.dumps({"clip": record.clip_id, "status": observed["clips"][record.clip_id]["status"]}), flush=True)
    if args.phase in ("decide", "both"):
        observed_status = json.loads((report / "observe.json").read_text())["clips"]
        decisions: list[dict[str, Any]] = []
        for record in records:
            if observed_status[record.clip_id]["status"] != "ok":
                decisions.append({"clip_id": record.clip_id, "observe_failed": observed_status[record.clip_id]})
                continue
            decisions.append(decide(runtime, dataset / record.path, report / "stores" / record.clip_id))
        document = {"schema": "court_side_clip_decisions_v1", "config_overrides": overrides,
                    "court_side": json_value(runtime.court_side), "summary": summarize([d for d in decisions if "detector" in d]),
                    "observe_failed": [d["clip_id"] for d in decisions if "observe_failed" in d], "clips": decisions}
        write_json_atomic(report / "decisions.json", document)
        print(json.dumps(document["summary"], indent=1))


if __name__ == "__main__":
    main()
