"""Cross-camera player association on labelled clips of a structured dataset.

``observe`` (GPU, through the shared training queue) runs court
detection/calibration and person detection, tracking and pose of the default
``pipeline.yaml`` into a run-owned store per clip. Ball, body and
reconstruction nodes are disabled; the side comes from the reviewed ball when
the association is evaluated. Each camera runs on its own so that one camera
whose tracking stops (for example ``person_capacity_exceeded``) is reported
with its evidence while the other cameras still produce tracks.
``observe.json`` lists, per clip and camera, the status and every track with
its observed frame count. ``sheets`` (CPU) renders, per clip and camera, a
contact sheet of every observed track (evenly spaced crops with their frame
index) from the stored tracks; it is the reviewing aid for the evaluation
labels. ``labels`` (CPU) turns a reviewed track assignment (``--review``, see
``tests/benchmarks/labels/player_association``) into tracker-independent box
labels, one JSON per clip under ``--labels-dir``. Nothing else is written
outside ``--report``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tasks.player_association.evaluation.labels import (
    ReviewedTrack,
    materialize,
    review_sha256,
)
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
from src.utils.video import OpenCVVideoFrameReader

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


def track_sheet(video: Path, tracks: PersonTrackingOutput, *, samples: int, crop_height: int) -> np.ndarray | None:
    """One row per track: ``samples`` evenly spaced observed crops labelled ``t<id> f<frame>``."""
    wanted: dict[int, list[tuple[int, int]]] = {}
    for row in range(len(tracks.track_ids)):
        frames = np.flatnonzero(tracks.observed[row])
        if not len(frames):
            continue
        for column, frame in enumerate(frames[np.linspace(0, len(frames) - 1, min(samples, len(frames))).round().astype(int)]):
            wanted.setdefault(int(frame), []).append((row, column))
    if not wanted:
        return None
    width = crop_height // 2
    label = 120
    rows = sorted({row for cells in wanted.values() for row, _ in cells})
    canvas: np.ndarray = np.full((len(rows) * (crop_height + 18), label + samples * (width + 4), 3), 32, np.uint8)
    for index, row in enumerate(rows):
        y = index * (crop_height + 18)
        text = f"t{int(tracks.track_ids[row])} n={int(tracks.observed[row].sum())}"
        cv2.putText(canvas, text, (4, y + crop_height // 2), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
    last = max(wanted)
    for packet in OpenCVVideoFrameReader(video, max_frames=last + 1):
        for row, column in wanted.get(packet.index, ()):
            x1, y1, x2, y2 = tracks.boxes_xyxy[row, packet.index]
            pad_x, pad_y = .1 * (x2 - x1), .05 * (y2 - y1)
            height, image_width = packet.frame.shape[:2]
            left, right = int(max(0, x1 - pad_x)), int(min(image_width, x2 + pad_x))
            top, bottom = int(max(0, y1 - pad_y)), int(min(height, y2 + pad_y))
            if right <= left or bottom <= top:
                continue
            crop = packet.frame[top:bottom, left:right]
            scale = min(crop_height / crop.shape[0], width / crop.shape[1])
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))))
            y = rows.index(row) * (crop_height + 18)
            x = label + column * (width + 4)
            canvas[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
            cv2.putText(canvas, f"f{packet.index}", (x, y + crop_height + 13), cv2.FONT_HERSHEY_SIMPLEX, .4, (200, 200, 0), 1)
    return canvas


def sheets(clip: Path, store_root: Path, output: Path, *, samples: int, crop_height: int) -> dict[str, str]:
    manifest, source = clip_source(clip)
    store = ClipStore(store_root, json_value(source))
    written: dict[str, str] = {}
    for video in source.videos:
        reference = store.active(f"person_tracking/{video.camera_id}")
        if reference is None:
            continue
        sheet = track_sheet(video.path, store.load(reference, ArtifactCodec(PersonTrackingOutput)), samples=samples, crop_height=crop_height)
        if sheet is None:
            continue
        path = output / f"{video.camera_id}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        written[video.camera_id] = str(path)
    return written


def labels(dataset: Path, report: Path, review_path: Path, output: Path) -> list[str]:
    """Materialize the reviewed clips of ``review_path`` from the observation stores under ``report``."""
    review = yaml.safe_load(review_path.read_text())
    observed = json.loads((report / "observe.json").read_text())
    written = []
    for clip_id, clip_review in review["clips"].items():
        if observed["clips"].get(clip_id, {}).get("status") != "ok":
            raise ValueError(f"{clip_id} has no completed observation in {report}")
        video_id, clip_name = clip_id.split("/")
        _, source = clip_source(dataset / "videos" / video_id / "clips" / clip_name)
        store = ClipStore(report / "stores" / clip_id, json_value(source))
        tracks: dict[str, list[ReviewedTrack]] = {}
        for camera, record in observed["clips"][clip_id]["cameras"].items():
            if record["status"] != "ok":
                raise ValueError(f"{clip_id} {camera} tracking stopped ({record['reason']}); it cannot be labelled")
            reference = store.active(f"person_tracking/{camera}")
            if reference is None:
                raise RuntimeError(f"{clip_id} person_tracking/{camera} has no adopted artifact")
            output_tracks = store.load(reference, ArtifactCodec(PersonTrackingOutput))
            tracks[camera] = [ReviewedTrack(int(track), output_tracks.boxes_xyxy[row], output_tracks.observed[row])
                              for row, track in enumerate(output_tracks.track_ids)]
        provenance = {"review": {"path": str(review_path.name), "sha256": review_sha256(review_path),
                                 "selection": clip_review["selection"]},
                      "observation": {"run": review["observe_run"], "code_sha256": observed["code_sha256"],
                                      "config_overrides": observed["config_overrides"]}}
        clip_labels = materialize(clip_id, source.num_frames, clip_review, tracks, provenance)
        path = output / f"{clip_id}.json"
        clip_labels.save(path)
        written.append(str(path))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="Root holding data/, ckpt/ and third_party/")
    parser.add_argument("--dataset", type=Path, required=True, help="Structured dataset directory (dataset.json)")
    parser.add_argument("--report", type=Path, required=True, help="Run-owned directory: player_association/evaluate/<experiment>/<run-id>")
    parser.add_argument("--phase", choices=("observe", "sheets", "labels"), default="observe")
    parser.add_argument("--review", type=Path, help="labels: reviewed track assignment (YAML)")
    parser.add_argument("--labels-dir", type=Path, help="labels: output directory (<video>/<clip>.json)")
    parser.add_argument("--samples", type=int, default=12, help="sheets: crops per track")
    parser.add_argument("--crop-height", type=int, default=160, help="sheets: crop height in pixels")
    parser.add_argument("--clip", action="append", default=[], help="Restrict to clip IDs (repeatable)")
    parser.add_argument("--override", action="append", default=[], help="Extra pipeline.yaml override")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    repo, dataset, report = args.repo.resolve(), args.dataset.resolve(), args.report.resolve()
    report.mkdir(parents=True, exist_ok=True)
    if args.phase == "labels":
        if args.review is None or args.labels_dir is None:
            parser.error("--phase labels requires --review and --labels-dir")
        print(json.dumps({"labels": labels(dataset, report, args.review.resolve(), args.labels_dir.resolve())}), flush=True)
        return
    manifest = load_dataset_manifest(dataset)
    records = [manifest.clips[key] for key in sorted(manifest.clips) if not args.clip or key in args.clip]
    if args.clip and len(records) != len(args.clip):
        raise ValueError(f"Unknown clip IDs: {sorted(set(args.clip) - {r.clip_id for r in records})}")
    if args.phase == "sheets":
        for record in records:
            written = sheets(dataset / record.path, report / "stores" / record.clip_id, report / "sheets" / record.clip_id,
                             samples=args.samples, crop_height=args.crop_height)
            print(json.dumps({"clip": record.clip_id, "sheets": written}), flush=True)
        return
    runtime, overrides = compose_runtime(repo, report, args.device, args.override, "observe")
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
