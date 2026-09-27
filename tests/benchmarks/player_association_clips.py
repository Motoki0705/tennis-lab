"""Cross-camera player association on labelled clips of a structured dataset.

``observe`` (GPU, through the shared training queue) runs court
detection/calibration and person detection, tracking and pose of the default
``pipeline.yaml`` into a run-owned store per clip. Ball, body and
reconstruction nodes are disabled; the side comes from the reviewed ball when
the association is evaluated. Each camera runs on its own so that one camera
whose tracking stops (for example ``person_capacity_exceeded``) is reported
with its evidence while the other cameras still produce tracks.
``observe.json`` lists, per clip and camera, the status and every track with
its observed frame count. Nothing is written outside ``--report``.
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
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.definition import standard_definition
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.tennis_scene.pipeline.runner import ComponentRunner
from src.tennis_scene.pipeline.source import build_clip_source
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec

CONFIG_DIR = Path(__file__).resolve().parents[2] / "src/tennis_scene/configs"


def compose_runtime(repo: Path, report: Path, device: str, overrides: list[str], name: str) -> tuple[PipelineRuntimeConfig, list[str]]:
    applied = [f"paths.project_root={CONFIG_DIR.parents[2]}", f"paths.data_root={repo / 'data'}",
        f"paths.checkpoint_root={repo / 'ckpt'}", f"paths.external_asset_root={repo / 'third_party'}",
        f"paths.artifact_root={report}", f"paths.output_root={report}", f"paths.cache_root={report / 'cache'}",
        f"device={device}", "output_directory=run", "ball_detection.enabled=false", "execution.court_side=load",
        "player_reconstruction.enabled=false", "ball_reconstruction.enabled=false", "gvhmr.enabled=false", *overrides]
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        config = compose(config_name="pipeline", overrides=applied)
    (report / f"{name}.pipeline_config.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    return PipelineRuntimeConfig.from_config(config, bind_inputs=False), applied


def clip_source(clip: Path) -> tuple[ClipManifest, ClipSource]:
    manifest = ClipManifest.load(clip)
    videos = tuple(manifest.media_path(camera) for camera in manifest.camera_ids)
    return manifest, build_clip_source(videos, tuple(manifest.camera_ids), clip_id=manifest.clip_id)


def observe(runtime: PipelineRuntimeConfig, code_identity: str, clip: Path, store_root: Path) -> dict[str, Any]:
    """Court calibration, then person detection, tracking and pose camera by camera."""
    _, source = clip_source(clip)
    nodes = standard_definition(runtime, source, code_identity=code_identity)
    store = ClipStore(store_root, json_value(source))
    ComponentRunner(nodes, store).run(targets=("court_calibration",))
    cameras: dict[str, Any] = {}
    for camera in source.camera_ids:
        runner = ComponentRunner(nodes, store)
        try:
            runner.run(targets=(f"pose_estimation/{camera}",))
        except ReconstructionUnavailable as stopped:
            cameras[camera] = {"status": "stopped", "reason": stopped.reason, "message": str(stopped),
                               "diagnostics": json_value(stopped.diagnostics), "seconds": runner.seconds}
            continue
        reference = store.active(f"person_tracking/{camera}")
        if reference is None:
            raise RuntimeError(f"person_tracking/{camera} completed without an adopted artifact")
        tracks = store.load(reference, ArtifactCodec(PersonTrackingOutput))
        cameras[camera] = {"status": "ok", "seconds": runner.seconds,
                           "tracks": [{"track_id": int(track), "observed_frames": int(tracks.observed[row].sum()),
                                       "source_track_ids": list(tracks.source_track_ids[row])}
                                      for row, track in enumerate(tracks.track_ids)],
                           "tracklet_links": len(tracks.tracklet_links)}
    return {"frames": source.num_frames, "cameras": cameras}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="Root holding data/, ckpt/ and third_party/")
    parser.add_argument("--dataset", type=Path, required=True, help="Structured dataset directory (dataset.json)")
    parser.add_argument("--report", type=Path, required=True, help="Run-owned directory: player_association/evaluate/<experiment>/<run-id>")
    parser.add_argument("--phase", choices=("observe",), default="observe")
    parser.add_argument("--clip", action="append", default=[], help="Restrict to clip IDs (repeatable)")
    parser.add_argument("--override", action="append", default=[], help="Extra pipeline.yaml override")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    repo, dataset, report = args.repo.resolve(), args.dataset.resolve(), args.report.resolve()
    report.mkdir(parents=True, exist_ok=True)
    runtime, overrides = compose_runtime(repo, report, args.device, args.override, "observe")
    manifest = load_dataset_manifest(dataset)
    records = [manifest.clips[key] for key in sorted(manifest.clips) if not args.clip or key in args.clip]
    if args.clip and len(records) != len(args.clip):
        raise ValueError(f"Unknown clip IDs: {sorted(set(args.clip) - {r.clip_id for r in records})}")
    code_identity = TennisSceneOrchestrator(runtime).code_identity
    observed: dict[str, Any] = {"schema": "player_association_observe_v1", "config_overrides": overrides,
                                "code_sha256": code_identity, "clips": {}}
    for record in records:
        try:
            observed["clips"][record.clip_id] = {"status": "ok", **observe(runtime, code_identity, dataset / record.path,
                                                                           report / "stores" / record.clip_id)}
        except (ReconstructionUnavailable, ValueError) as error:
            # A clip whose court cannot be calibrated is reported, never skipped silently.
            observed["clips"][record.clip_id] = {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                                                 "reason": getattr(error, "reason", None)}
        write_json_atomic(report / "observe.json", observed)
        print(json.dumps({"clip": record.clip_id, "status": observed["clips"][record.clip_id]["status"]}), flush=True)


if __name__ == "__main__":
    main()
