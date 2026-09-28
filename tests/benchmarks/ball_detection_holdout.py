"""Paired Meiji holdout: CPU preflight, queued inference, and CPU-only rescoring.

See tests/benchmarks/README.md. Inputs and pose stores are read-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, fields, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from src.tasks.ball_detection.data.store import BallFrameStore, ClipRecord
from src.tasks.ball_detection.evaluation.holdout_inference import (
    FramePredictions,
    predict_store_clip,
)
from src.tasks.ball_detection.evaluation.holdout_metrics import (
    HoldoutReferences,
    clip_references,
    summarize_holdout,
    wrist_distances,
)
from src.tasks.ball_detection.inference.checkpoint import load_ball_checkpoint
from src.tasks.ball_detection.inference.predictor import BallDetectionPredictor
from src.tasks.ball_detection.model_io.contracts import BallCandidateConfig
from src.tennis_scene.pipeline.artifacts import document_digest, write_json_atomic
from src.tennis_scene.pipeline.observation_types import ObjectObservations
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.tennis_scene.pipeline.storage.scene_index import read_component_descriptor


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def pose_reference(
    reference: HoldoutReferences, clip: ClipRecord, poses: Path, *, threshold: float,
) -> tuple[HoldoutReferences, dict[str, Any]]:
    """Use only the explicitly indexed pose; absence is a recorded unknown stratum."""
    parts = clip.clip_id.split("/")
    if len(parts) != 4 or parts[0] != "meiji" or parts[1] != clip.group_id or parts[3] != clip.camera_id:
        raise ValueError(f"Unexpected Meiji store identity: {clip.clip_id}")
    root = poses / parts[1] / parts[2]
    index = root / "scene.json"
    record: dict[str, Any] = {"clip_id": clip.clip_id, "scene_index": str(index)}
    if not index.is_file():
        return reference, dict(record, status="missing_scene", known_distance_frames=0)
    document = json.loads(index.read_text())
    if (document["schema"] != "tennis_scene_index_v1"
            or document_digest(document["source"]) != document["source_sha256"]):
        raise ValueError(f"Invalid pose scene source: {index}")
    source = document["source"]
    videos = [v for v in source["videos"] if v["camera_id"] == clip.camera_id]
    if source["clip_id"] != "/".join(parts[1:3]) or len(videos) != 1:
        raise ValueError(f"Pose clip/camera identity mismatch: {clip.clip_id}")
    video = videos[0]
    expected = (clip.media_sha256, clip.source_width, clip.source_height, clip.frame_count)
    actual = (video["sha256"], video["width"], video["height"], video["num_frames"])
    if actual != expected or not np.isclose(video["fps"], float(Fraction(clip.fps)), rtol=1e-6):
        raise ValueError(f"Pose media/size/timeline mismatch: {clip.clip_id}")
    node = f"pose_estimation/{clip.camera_id}"
    artifact = document["artifacts"].get(node)
    record["scene_index_sha256"] = sha256(index)
    if artifact is None:
        return reference, dict(record, status="missing_pose", known_distance_frames=0)
    if (artifact["schema"], artifact["version"]) != ("person_poses", 1):
        raise ValueError("Unsupported pose artifact schema")
    descriptor = read_component_descriptor(
        root, artifact, node=node, source_sha256=document["source_sha256"],
    )
    pose = ArtifactCodec(ObjectObservations).load(
        descriptor["payload"], (root / artifact["path"]).parent, descriptor["arrays"],
    )
    if (pose.camera_ids != (clip.camera_id,) or pose.num_frames != clip.frame_count
            or pose.size != (clip.source_width, clip.source_height)
            or pose.uv_px.shape[-2] != 17
            or not np.isclose(pose.fps, float(Fraction(clip.fps)), rtol=1e-6)):
        raise ValueError(f"Pose observation axes mismatch: {clip.clip_id}")
    # np.take preserves (T,D,2 wrists,2) instead of advanced indexing's axis move.
    wrists = np.take(pose.uv_px[0], [9, 10], axis=2).reshape(clip.frame_count, -1, 2)
    valid = np.take(pose.visibility(threshold)[0], [9, 10], axis=2).reshape(clip.frame_count, -1)
    distance = wrist_distances(reference.uv, wrists, valid)
    record.update(status="loaded", artifact=artifact, wrist_supported_frames=int(valid.any(axis=1).sum()),
                  known_distance_frames=int(np.isfinite(distance).sum()))
    return replace(reference, wrist_distance=distance), record


def preflight(args: argparse.Namespace) -> tuple[BallFrameStore, tuple[ClipRecord, ...], HoldoutReferences, dict[str, Any]]:
    store = BallFrameStore(args.store.resolve())
    clips = store.split_clips("test", ["meiji"])
    if not clips or {c.group_id for c in clips} != {args.video}:
        raise ValueError("The entire Meiji test split must match the declared holdout video")
    if any(c.source == "meiji" and c.group_id == args.video and c.split != "test" for c in store.clips):
        raise ValueError("Holdout video also appears in train/validation")
    if sum(c.frame_count for c in clips) != args.expected_frames or len(clips) != args.expected_clips:
        raise ValueError("Meiji holdout coverage differs from the declared frame/clip counts")
    if not args.poses.is_dir():
        raise FileNotFoundError(f"Declared pose store root does not exist: {args.poses}")
    records, pose_records = [], []
    for clip in clips:
        reference, pose_record = pose_reference(
            clip_references(store, clip), clip, args.poses.resolve(), threshold=args.wrist_confidence,
        )
        records.append(reference)
        pose_records.append(pose_record)
    combined = HoldoutReferences(**{
        field.name: np.concatenate([getattr(r, field.name) for r in records])
        for field in fields(HoldoutReferences)
    })
    checkpoints: dict[str, Any] = {}
    for name, path in (("ft_e13", args.baseline), ("mixed_ft", args.treatment)):
        loaded = load_ball_checkpoint(path.resolve())
        contract = {
            "model": OmegaConf.to_container(loaded.config.model, resolve=True),
            "image_size": list(loaded.config.data.image_size),
            "normalization": asdict(loaded.image_normalization),
        }
        if checkpoints and contract != checkpoints["ft_e13"]["contract"]:
            raise ValueError("Paired checkpoints have different architecture/input contracts")
        checkpoints[name] = {"path": str(path.resolve()), "sha256": sha256(path), "contract": contract}
        del loaded
    if not 1 <= args.stride <= int(checkpoints["ft_e13"]["contract"]["model"]["num_frames"]):
        raise ValueError("Stride must cover the complete timeline")
    if (not np.isfinite([args.score_threshold, args.distance_px, args.near_wrist_px, args.wrist_confidence]).all()
            or not 0 <= args.score_threshold <= 1 or not 0 <= args.wrist_confidence <= 1
            or min(args.distance_px, args.near_wrist_px, args.batch_size) <= 0):
        raise ValueError("Invalid evaluation thresholds/batch size")
    protocol = {
        "schema": "ball_meiji_holdout_v1", "store": str(store.directory),
        "store_sha256": {name: sha256(store.directory / name) for name in ("metadata.json", "index.npz")},
        "clips": [asdict(c) for c in clips], "frames": len(combined.row),
        "checkpoints": checkpoints, "poses": pose_records,
        "decode": {"stride": args.stride, "batch_size": args.batch_size,
                   "tail_policy": "backfill", "overlap": "max_score_later_tie", "subpixel_refine": True,
                   "candidates": asdict(BallCandidateConfig()), "trajectory_gate": False,
                   "coordinate_conversion": "normalized * (stored_WH - 1) / store_scale",
                   "resize": "stored JPEG -> INTER_LINEAR -> checkpoint image_size"},
        "metrics": {"score_threshold": args.score_threshold, "distance_px": args.distance_px,
                    "near_wrist_px": args.near_wrist_px},
        "wrist_confidence": args.wrist_confidence,
        "runtime": {"torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__,
                    "device": args.device, "precision": "float32", "tf32": False},
    }
    return store, clips, combined, protocol


def summarize(report: Path) -> None:
    """Recompute all tables from saved arrays without reading data/models/poses."""
    protocol = json.loads((report / "protocol.json").read_text())
    manifest = json.loads((report / "predictions.json").read_text())
    if protocol["schema"] != "ball_meiji_holdout_v1" or set(manifest) != set(protocol["checkpoints"]):
        raise ValueError("Incomplete paired prediction manifest")
    ref_path = report / "references.npz"
    if sha256(ref_path) != protocol["references_sha256"]:
        raise ValueError("Reference checksum mismatch")
    with np.load(ref_path, allow_pickle=False) as data:
        reference = HoldoutReferences(**dict(data))
    if len(reference.row) != protocol["frames"]:
        raise ValueError("Saved reference coverage mismatch")
    result: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for name, entry in manifest.items():
        path = report / entry["file"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Prediction checksum mismatch: {name}")
        with np.load(path, allow_pickle=False) as data:
            if not np.array_equal(data["row"], reference.row):
                raise ValueError("Predictions changed source frame identity/order")
            prediction = FramePredictions(**{k: data[k] for k in data.files if k != "row"})
        result[name] = summarize_holdout(reference, prediction, **protocol["metrics"])
        rows.extend({"model": name, "stratum": key, **value} for key, value in result[name].items())
    write_json_atomic(report / "metrics.json", result)
    with (report / "metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def number(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.4f}"
    lines = ["# Meiji holdout比較", "",
             "主指標はoverall_observedとcamera行。visibility行は位置推定ラベルも含む参考値。",
             f"手首{protocol['metrics']['near_wrist_px']}pxは打球/飛行のproxy。pose欠落はunknownで、既存pose部分集合への選択バイアスがある。",
             f"score >= {protocol['metrics']['score_threshold']}、一致距離 <= {protocol['metrics']['distance_px']} source px。",
             "accepted p95はscore閾値を通る全位置誤差（一致距離超も含む）。raw p95は低scoreも含むargmax誤差。",
             "注釈位置不明は負例にせず、recallの分母から除外する。詳細条件・hashはprotocol.json。", "",
             "| model | stratum | frames | references | missing | wrong | recall | top-K recall | accepted p95 px | raw p95 px |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['stratum']} | {row['frames']} | {row['reference_frames']} | "
                     f"{row['missing_reference_frames']} | {row['wrong_reference_frames']} | {number(row['recall'])} | "
                     f"{number(row['topk_recall_unthresholded'])} | {number(row['accepted_p95_px'])} | {number(row['raw_argmax_p95_px'])} |")
    (report / "comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({name: values["overall_observed"] for name, values in result.items()}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "infer", "summarize"), default="infer")
    parser.add_argument("--store", type=Path)
    parser.add_argument("--poses", type=Path, help="Existing #933 stores root (video/clip/scene.json)")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--treatment", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--video", default="video_001")
    parser.add_argument("--expected-frames", type=int, default=36006)
    parser.add_argument("--expected-clips", type=int, default=63)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--distance-px", type=float, default=20.0)
    parser.add_argument("--near-wrist-px", type=float, default=100.0)
    parser.add_argument("--wrist-confidence", type=float, default=0.5)
    args = parser.parse_args()
    report = args.report.resolve()
    if args.phase == "summarize":
        summarize(report)
        return
    if any(getattr(args, name) is None for name in ("store", "poses", "baseline", "treatment")):
        parser.error("preflight/infer require --store, --poses, --baseline and --treatment")
    if (report / "protocol.json").exists():
        raise FileExistsError("Use a new report directory; completed/partial evidence is immutable")
    report.mkdir(parents=True, exist_ok=True)
    store, clips, reference, protocol = preflight(args)
    np.savez_compressed(report / "references.npz", **asdict(reference))
    protocol["references_sha256"] = sha256(report / "references.npz")
    write_json_atomic(report / "protocol.json", protocol)
    print(f"Validated {len(clips)} clips / {len(reference.row)} unique frames", flush=True)
    if args.phase == "preflight":
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    manifest: dict[str, Any] = {}
    for name, entry in protocol["checkpoints"].items():
        checkpoint = Path(entry["path"])
        if sha256(checkpoint) != entry["sha256"]:
            raise ValueError("Checkpoint changed after preflight")
        loaded = load_ball_checkpoint(checkpoint)
        predictor = BallDetectionPredictor(
            loaded.model_io, torch.device(args.device), subpixel_refine=True,
            image_normalization=loaded.image_normalization,
        )
        predictions = []
        for i, clip in enumerate(clips):
            predictions.append(predict_store_clip(
                store, clip, predictor, image_size=tuple(entry["contract"]["image_size"]),
                stride=args.stride, batch_size=args.batch_size, candidates=BallCandidateConfig(),
            ))
            print(f"{name} {i + 1}/{len(clips)} {clip.clip_id} frames={clip.frame_count}", flush=True)
        merged = FramePredictions(**{
            field.name: np.concatenate([getattr(p, field.name) for p in predictions])
            for field in fields(FramePredictions)
        })
        path = report / f"{name}.npz"
        np.savez_compressed(path, row=reference.row, **asdict(merged))
        manifest[name] = {"file": path.name, "sha256": sha256(path)}
        write_json_atomic(report / "predictions.json", manifest)
        del predictor, loaded, predictions, merged
        if args.device == "cuda":
            torch.cuda.empty_cache()
    summarize(report)


if __name__ == "__main__":
    main()
