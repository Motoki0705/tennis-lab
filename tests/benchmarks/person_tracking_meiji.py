"""Fixed 4-camera Meiji re-evaluation of tracking runs 8–10."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from person_tracking_meiji_features import (  # type: ignore[import-not-found]
    FeatureProducer,
    weight_path,
)
from person_tracking_meiji_inputs import (  # type: ignore[import-not-found]
    CLIP,
    ENCODERS,
    KPR,
    adopted_clip,
    feature_path,
    load_detections,
    verify_digest,
)
from person_tracking_meiji_replay import (  # type: ignore[import-not-found]
    BASELINE,
    DEFAULT,
    HISTORICAL,
    HYBRID,
    NATIVE_ELIGIBLE,
    replay_clip,
)
from person_tracking_meiji_score import (  # type: ignore[import-not-found]
    ASSOCIATION_CONFIG,
    SCORE_STRATA,
    baseline,
    calibration_errors,
    candidate,
    label_strata,
)

from src.tasks.person_tracking.comparison import pool_scores, score_clip
from src.tasks.person_tracking.strongsort_offline import AF_WEIGHT_SHA256, AFLink
from src.tasks.player_association.evaluation.dataset_labels import label_path
from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tasks.player_association.pseudo_annotation.extract import load_plan
from src.tasks.player_association.pseudo_annotation.frozen_run import FrozenRun
from src.utils.checksum import dual_sha256
from src.utils.io import write_json_atomic

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "knowledge/runs/run-i2443-meiji-4cam-tracker-comparison/README.md"
TRACKERS = (BASELINE, *HISTORICAL, *(v.replace(CLIP, KPR) for v in NATIVE_ELIGIBLE[1:]))


def _commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _identity(frozen: FrozenRun, args: argparse.Namespace) -> str:
    tracking_code = [
        *sorted((ROOT / "src/tasks/person_tracking").glob("*.py")),
        *sorted((ROOT / "src/tasks/person_tracking/deep_ocsort_vendor").glob("*.py")),
        *sorted((ROOT / "src/tasks/player_association/appearance").rglob("*.py")),
        ROOT / "src/tennis_scene/pipeline/components/tracking_identity.py",
        *(ROOT / "tests/benchmarks" / f"{name}.py" for name in (
            "person_tracking_linking", "person_tracking_meiji_replay",
            "person_tracking_meiji_features", "person_tracking_meiji_inputs", "person_kpr_features")),
    ]
    # Metric changes do not invalidate an existing tracking archive.
    tracking_code = [path for path in tracking_code if path.stem not in ("comparison", "projection", "evaluation")]
    document = {"schema_version": 1, "snapshot": frozen.sha256, "selection": dual_sha256(args.selection),
                "saved_clip": args.saved_clip, "aflink": AF_WEIGHT_SHA256,
                "versions": {"numpy": np.__version__, "opencv": cv2.__version__, "torch": str(torch.__version__)},
                "code": {str(p.relative_to(ROOT)): dual_sha256(p) for p in tracking_code}}
    path = args.report / "tracking-identity.json"
    if path.exists() and json.loads(path.read_text()) != document:
        raise ValueError("Tracking inputs changed. Use a separate report directory.")
    if not path.exists():
        write_json_atomic(path, document)
    return str(dual_sha256(path))


def preflight(args: argparse.Namespace, frozen: FrozenRun, clips: list[str]) -> None:
    if args.adopted_dataset is None:
        raise ValueError("Preflight needs the adopted #2363 dataset")
    weights = {encoder: str(weight_path(args.checkpoint_root, encoder)) for encoder in ENCODERS}
    aflink = verify_digest(args.checkpoint_root / "person_tracking/AFLink_epoch20.pth", AF_WEIGHT_SHA256)
    results = {}
    for clip_id in clips:
        clip = frozen.clip(clip_id)
        adopted = adopted_clip(frozen, clip, args.adopted_dataset)
        if set(clip.local_cameras()) != set(clip.clip.camera_ids):
            raise ValueError(f"Missing local calibration for {clip_id}")
        rows = {}
        for camera in clip.clip.camera_ids:
            data = load_detections(adopted, camera)
            adopted.store.descriptor(f"person_appearance_features/{camera}")
            rows[camera] = len(data.rows)
            extent = np.rint(data.boxes).clip([0, 0, 0, 0], [clip.clip.width, clip.clip.height] * 2)
            if (extent[:, 2:] <= extent[:, :2]).any():
                raise ValueError(f"KPR has a zero-area rounded crop for {clip_id}/{camera}")
        results[clip_id] = {"frames": clip.clip.num_frames, "rows": rows,
                            "scene_sha256": dual_sha256(adopted.store.index_path), "saved_status": clip.outcome["status"]}
    write_json_atomic(args.report / "preflight.json", {
        "commit": _commit(), "protocol_sha256": dual_sha256(PROTOCOL), "snapshot_sha256": frozen.sha256,
        "selection_sha256": dual_sha256(args.selection), "weights": weights, "aflink": str(aflink), "clips": results,
    })
    frames = sum(c["frames"] for c in results.values())
    total_rows = sum(sum(c["rows"].values()) for c in results.values())
    print(f"Verified {len(results)} clips, {frames} synced frames, {total_rows} detection rows", flush=True)


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _score(args: argparse.Namespace, frozen: FrozenRun, plan: dict[str, Any], clips: list[str]) -> None:
    if args.labels_dataset is None or args.scope is None:
        raise ValueError("Scoring needs --scope and --labels-dataset")
    variants = args.variant or (BASELINE, *HISTORICAL)
    identity = _identity(frozen, args) if any(variant != BASELINE for variant in variants) else None
    calibration = calibration_errors(args.annotation_errors, frozen.sha256, dual_sha256(args.selection))
    required = args.scope == "full"
    per_condition: dict[str, list[dict[str, Any]]] = {}
    label_records = {}
    for clip_id in clips:
        path = label_path(args.labels_dataset, clip_id)
        if not required and not path.exists():
            continue
        clip = frozen.clip(clip_id)
        labels = ClipLabels.load(path)
        strata = label_strata(labels, clip, frozen, plan, require_full=required)
        label_records[clip_id] = {"sha256": dual_sha256(path), "strata": strata,
                                  "boxes": sum(len(c.frames) for c in labels.cameras.values()),
                                  "ambiguous_boxes": sum(int((c.person_index < 0).sum()) for c in labels.cameras.values())}
        local = clip.local_cameras()
        for variant in variants:
            if variant == BASELINE:
                outcomes = {CLIP: baseline(clip, local)}
            else:
                if args.adopted_dataset is None or identity is None:
                    raise ValueError("Candidate scoring needs the adopted #2363 dataset and tracking identity")
                adopted = adopted_clip(frozen, clip, args.adopted_dataset)
                names = (CLIP,) if variant in (DEFAULT, HYBRID) else args.association_encoder or ENCODERS
                outcomes = candidate(frozen, clip, adopted, local, args.report, variant, names, strata["all"], identity,
                                     saved_clip=args.saved_clip)
            for encoder, (prediction, evidence) in outcomes.items():
                condition = f"{variant}/{encoder}"
                per_condition.setdefault(condition, []).append({
                    "clip": clip_id, "evidence": evidence, "strata": score_clip(labels, prediction, strata),
                })
                print(f"Scored {condition} {clip_id}, {len(strata['all'])} frames", flush=True)
    if not label_records:
        raise ValueError("No labels were scored")
    table: dict[str, dict[str, Any]] = {}
    csv_rows = []
    for condition, records in per_condition.items():
        table[condition] = {}
        for stratum in SCORE_STRATA:
            scores = pool_scores(r["strata"][stratum] for r in records if stratum in r["strata"])
            audit = calibration.get(stratum)
            table[condition][stratum] = {**scores, "annotation_error": audit}
            csv_rows.append({
                "condition": condition, "stratum": stratum, "frames": scores["frames"], "clips": scores["clips"],
                "stopped_clips": scores["stopped_clips"], "labelled_camera_units": scores["raw"].get("player_units", 0),
                **{f"{stage}_{k}": scores[stage].get(k, 0)
                   for stage in ("raw", "group") for k in ("idf1", "id_switches", "fragments")},
                "selection_precision": scores["selection_precision"], "selection_recall": scores["selection_recall"],
                "pair_f1": scores["pairs"]["f1"], "matched_decided_pair_f1": scores["matched_decided_pairs"]["f1"],
                "3d_coverage": scores["coverage_3d"], "valid_player_frames": scores["valid_player_frames"],
                "labelled_player_frames": scores["labelled_player_frames"],
                "checked_boxes": audit["checked_boxes"] if audit is not None else None,
                "annotation_box_error_rate": audit["box_error_rate"] if audit is not None else None,
            })
    ranking = {name: table[f"{name}/{CLIP}"]["all"]["raw"] for name in NATIVE_ELIGIBLE if f"{name}/{CLIP}" in table}
    selected = None
    if required and len(ranking) == len(NATIVE_ELIGIBLE) and all(r["idf1"] is not None for r in ranking.values()):
        best = max(r["idf1"] for r in ranking.values())
        eligible = (name for name, counts in ranking.items() if best - counts["idf1"] <= 1e-6)
        selected = min(eligible, key=lambda name: (ranking[name]["id_switches"], ranking[name]["fragments"], name))
    args.report.mkdir(parents=True, exist_ok=True)
    _csv(args.report / f"{args.scope}-comparison.csv", csv_rows)
    write_json_atomic(args.report / f"{args.scope}-scores.json", {
        "schema_version": 1, "commit": _commit(), "scope": args.scope, "snapshot_sha256": frozen.sha256,
        "saved_clip": args.saved_clip, "tracking_identity_sha256": identity,
        "association_config_sha256": dual_sha256(ASSOCIATION_CONFIG), "selection_sha256": dual_sha256(args.selection),
        "annotation_audit_sha256": None if args.annotation_errors is None else dual_sha256(args.annotation_errors),
        "label_records": label_records, "tables": table, "clip_results": per_condition, "run9_kpr_selection": selected,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "features", "track", "score"), required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--adopted-dataset", type=Path)
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "ckpt")
    parser.add_argument("--clip", action="append")
    parser.add_argument("--variant", choices=TRACKERS, action="append")
    parser.add_argument("--encoder", choices=ENCODERS)
    parser.add_argument("--association-encoder", choices=ENCODERS, action="append")
    parser.add_argument("--saved-clip", action="store_true", help="Use #2363 CLIP for a tracking/scoring smoke test")
    parser.add_argument("--scope", choices=("pilot", "full"))
    parser.add_argument("--labels-dataset", type=Path)
    parser.add_argument("--annotation-errors", type=Path)
    args = parser.parse_args()
    if args.saved_clip and (args.phase not in ("track", "score") or args.scope == "full"):
        parser.error("--saved-clip is reserved for a smoke test in its own report directory")
    started = time.perf_counter()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    frozen = FrozenRun(args.snapshot)
    plan = load_plan(args.selection, frozen)
    planned = {name for name, record in plan["clips"].items() if record["sampled_frames"]}
    clips = sorted(planned if args.clip is None else set(args.clip))
    if not clips or set(clips) - planned or (args.scope == "full" and set(clips) != planned):
        raise ValueError("Clip list must cover the requested portion of the frozen plan")
    if args.phase == "score":
        _score(args, frozen, plan, clips)
    elif args.phase == "preflight":
        preflight(args, frozen, clips)
    else:
        if args.adopted_dataset is None:
            parser.error("Feature extraction and tracking require --adopted-dataset")
        if args.phase == "features":
            if args.encoder is None or not {"TENNIS_REPRO_DIR", "TENNIS_RUN_ID"} <= os.environ.keys():
                parser.error("Run --phase features with an explicit --encoder through the shared training queue")
            repro = Path(os.environ["TENNIS_REPRO_DIR"])
            (repro / "output_dir.txt").write_text(str(args.report.resolve()) + "\n")
            torch.backends.cuda.matmul.allow_tf32 = False
            identity = _identity(frozen, args)
            producer = FeatureProducer(args.encoder, args.checkpoint_root)
            manifests = {}
        else:
            variants = tuple(args.variant or HISTORICAL)
            if BASELINE in variants:
                parser.error("The saved default is already available from the snapshot")
            identity = _identity(frozen, args)
            aflink = AFLink(args.checkpoint_root / "person_tracking/AFLink_epoch20.pth")
        for clip_id in clips:
            frozen_clip = frozen.clip(clip_id)
            adopted = adopted_clip(frozen, frozen_clip, args.adopted_dataset)
            if args.phase == "features":
                for camera in frozen_clip.clip.camera_ids:
                    producer.extract(frozen, adopted, camera, args.report)
                    receipt = feature_path(args.report, args.encoder, clip_id, camera).with_suffix(".json")
                    manifests[f"{clip_id}/{camera}"] = dual_sha256(receipt)
            else:
                replay_clip(frozen, frozen_clip, adopted, variants, args.report, aflink, identity,
                            saved_clip=args.saved_clip)
        if args.phase == "features" and args.clip is None:
            write_json_atomic(args.report / "features" / args.encoder / "manifest.json", {
                "encoder": args.encoder, "commit": _commit(), "identity_sha256": identity,
                "run_id": os.environ["TENNIS_RUN_ID"], "receipts": manifests,
                "elapsed_seconds": time.perf_counter() - started,
            })
    print(f"Completed {args.phase} in {time.perf_counter() - started:.2f}s", flush=True)


if __name__ == "__main__":
    main()
