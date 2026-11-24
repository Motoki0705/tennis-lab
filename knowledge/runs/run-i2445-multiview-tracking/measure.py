"""Replay fixed Meiji inputs, publish doubles tracks and score the singles labels."""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from evaluate_projection import evaluate_clip, summarize_views
from probe import read_json
from replay import DEV_CLIPS, _read_human_eval_inputs
from score_holdout import REPAIRED_CLIP, _label_pins, _observations

from src.tasks.person_tracking.evaluation import score_units
from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tennis_scene.multiview_pose.reconstruct import DEFAULT_CONFIG, reconstruct
from src.tennis_scene.multiview_pose.schema import NODE as C_NODE
from src.tennis_scene.multiview_tracking.schema import (
    PoseTracks,
    SampleEvent,
    TrackEnd,
    TrackingConfig,
)
from src.tennis_scene.multiview_tracking.storage import read_tracks, track_store
from src.tennis_scene.multiview_tracking.tracking import track, track_with_diagnostics
from src.tennis_scene.pipeline.artifacts import document_digest, json_value
from src.tennis_scene.pipeline.source_identity import source_digests
from src.utils.checksum import dual_sha256
from src.utils.io import write_json_atomic

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
C_EVIDENCE = RUN.parent / "run-multiview-pose-link-recall-20261124"
INTEGRATION_EVIDENCE = RUN.parent / "run-i2445-player-slot-integration"
SINGLES = TrackingConfig(players_per_side=1)
DOUBLES = TrackingConfig(players_per_side=2)


def _clip_diagnostics(result: PoseTracks) -> dict[str, Any]:
    frames = np.repeat(
        np.arange(len(result.frame_offsets) - 1), np.diff(result.frame_offsets)
    )
    measured = result.observed
    selected_ids = result.selection.selected_track_ids
    selected = np.isin(result.track_ids, selected_ids)
    threshold = result.selection.players_per_side * 2
    real_count = np.bincount(
        frames[selected & measured], minlength=len(result.frame_offsets) - 1
    )
    all_count = np.bincount(frames[selected], minlength=len(result.frame_offsets) - 1)
    return {
        "source_frames": len(result.frame_offsets) - 1,
        "tracks": len(result.end_frames),
        "events": {
            event.name.lower(): int((result.event == event).sum())
            for event in SampleEvent
        },
        "track_endings": {
            reason.name.lower(): int((result.end_reasons == reason).sum())
            for reason in TrackEnd
        },
        "frame_ambiguous_observations": int(result.frame_ambiguous.sum()),
        "temporal_ambiguous_observations": int((result.id_ambiguous & measured).sum()),
        "temporal_ambiguous_predictions": int((result.id_ambiguous & ~measured).sum()),
        "selected_track_ids": list(selected_ids),
        "selected_root_median_xy_m": [
            np.median(
                result.torso_m[measured & (result.track_ids == i), :2], axis=0
            ).tolist()
            for i in selected_ids
        ],
        "selected_diagnostics": [
            json_value(result.selection.decisions[i]) for i in selected_ids
        ],
        "missing_per_side": list(result.selection.missing_per_side),
        "selected_measured_frames": int((selected & measured).sum()),
        "selected_predicted_frames": int((selected & ~measured).sum()),
        "full_lineup_measured_frames": int((real_count == threshold).sum()),
        "full_lineup_available_frames": int((all_count == threshold).sum()),
    }


def _summarize(clips: dict[str, dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    return {
        "clips": keys,
        **{
            method: summarize_views(
                [view for clip in keys for view in clips[clip][method].values()]
            )
            for method in ("projected", "supporting_rows")
        },
    }


def _score_clip(inputs, labels, result, pins, seconds):
    views = evaluate_clip(inputs, labels, result)
    clip = {
        "inputs": pins,
        "tracking_seconds": seconds,
        **_clip_diagnostics(result),
        **{
            method: {
                "all_cameras": summarize_views(list(cameras.values())),
                "cameras": {
                    camera: summarize_views([view]) for camera, view in cameras.items()
                },
            }
            for method, cameras in views.items()
        },
    }
    return clip, views


def development(dataset: Path, archive: Path) -> dict[str, Any]:
    clips, view_sets, baseline_view_sets = {}, {}, {}
    legacy = read_json(RUN / "meiji3.json")
    for clip in DEV_CLIPS:
        inputs, labels, pins = _read_human_eval_inputs(dataset, archive, clip)
        candidates = reconstruct(inputs)
        start = perf_counter()
        result, integration = track_with_diagnostics(
            candidates, fps=inputs.source.fps, config=SINGLES
        )
        row, view_sets[clip] = _score_clip(
            inputs, labels, result, pins, perf_counter() - start
        )
        first_stage = track(
            candidates,
            fps=inputs.source.fps,
            config=replace(SINGLES, fragment_integration=None),
        )
        baseline_view_sets[clip] = evaluate_clip(inputs, labels, first_stage)
        baseline = _summarize(baseline_view_sets, [clip])
        if any(
            baseline[method] != legacy["clips"][clip][method]["all_cameras"]
            for method in ("projected", "supporting_rows")
        ):
            raise ValueError(
                f"The first-stage comparison differs from the pinned result: {clip}"
            )
        clips[clip] = {
            **row,
            "first_stage": {
                "missing_per_side": list(first_stage.selection.missing_per_side),
                "selected_track_ids": list(first_stage.selection.selected_track_ids),
                **baseline,
            },
            "fragment_integration": json_value(integration),
            "switch_changes": _switch_changes(
                baseline_view_sets[clip], view_sets[clip]
            ),
        }
        print(clip, "scored", flush=True)
    return {
        "config": json_value(SINGLES),
        "clips": clips,
        "first_stage_overall": _summarize(baseline_view_sets, list(DEV_CLIPS)),
        "overall": _summarize(view_sets, list(DEV_CLIPS)),
    }


def _switch_changes(first_stage, integrated) -> dict[str, Any]:
    def switch_frames(views, key):
        units = [unit for view in views.values() for unit in view[key]]
        return {
            (event["camera"], event["frame"], event["person"])
            for event in score_units(units)["switch_events"]
        }

    changes = {}
    for mode in first_stage:
        changes[mode] = {}
        for name, key in (
            ("selected_id", "units"),
            ("all_player_id", "all_player_units"),
        ):
            before = switch_frames(first_stage[mode], key)
            after = switch_frames(integrated[mode], key)
            changes[mode][name] = {
                "appeared": sorted(after - before),
                "removed": sorted(before - after),
            }
    return changes


def doubles(candidate_stores: Path, output: Path) -> dict[str, Any]:
    clips = {}
    legacy = read_json(RUN / "meiji4.json")
    for clip, pinned in sorted(read_json(C_EVIDENCE / "meiji4.json")["clips"].items()):
        origin = candidate_stores / clip
        expected = read_json(origin / "scene.json")["artifacts"][C_NODE]
        if (expected["artifact_id"], expected["sha256"]) != (
            pinned["artifact_id"],
            pinned["descriptor_sha256"],
        ):
            raise ValueError(
                f"{clip} needs the fixed input from the #2444 repeated benchmark"
            )
        destination = output / clip
        start = perf_counter()
        ref = track_store(
            origin, destination, DOUBLES, expected_artifact_id=pinned["artifact_id"]
        )
        result = read_tracks(destination, expected_artifact_id=ref.artifact_id)
        descriptor = read_json(destination / ref.path)
        if descriptor["provenance"]["frame_pose_candidates"] != expected:
            raise ValueError(f"{clip} published a different input revision")
        row = _clip_diagnostics(result)
        baseline = legacy["clips"][clip]
        guarded = ("full_lineup_measured_frames", "full_lineup_available_frames")
        if row["missing_per_side"] != baseline["missing_per_side"] or any(
            row[key] < baseline[key] for key in guarded
        ):
            raise ValueError(
                f"Player slot integration reduced doubles lineup coverage: {clip}"
            )
        slots = descriptor["provenance"]["fragment_integration"]["slots"]
        joins = [join for slot in slots for join in slot["joins"]]
        clips[clip] = {
            "candidate_artifact_id": expected["artifact_id"],
            "candidate_descriptor_sha256": expected["sha256"],
            "tracking_artifact_id": ref.artifact_id,
            "tracking_descriptor_sha256": ref.sha256,
            "tracking_and_store_seconds": perf_counter() - start,
            "first_stage": {key: baseline[key] for key in guarded},
            "fragment_joins": len(joins),
            "long_fragment_joins": sum(join["long_gap"] for join in joins),
            "ambiguous_fragment_joins": sum(join["ambiguous"] for join in joins),
            **row,
        }
        print(
            clip,
            row["full_lineup_measured_frames"],
            "/",
            row["source_frames"],
            flush=True,
        )
    return {
        "config": json_value(DOUBLES),
        "clips": clips,
        "overall": {
            "clips": len(clips),
            "full_lineup_clips": sum(
                row["missing_per_side"] == [0, 0] for row in clips.values()
            ),
            "first_stage": {
                key: legacy["overall"][key]
                for key in (
                    "full_lineup_clips",
                    "full_lineup_measured_frames",
                    "full_lineup_available_frames",
                )
            },
            **{
                key: sum(row[key] for row in clips.values())
                for key in (
                    "source_frames",
                    "full_lineup_measured_frames",
                    "full_lineup_available_frames",
                    "selected_measured_frames",
                    "selected_predicted_frames",
                    "fragment_joins",
                    "long_fragment_joins",
                    "ambiguous_fragment_joins",
                )
            },
        },
    }


def reserved_protocol() -> dict[str, Any]:
    return {
        "config": json_value(SINGLES),
        "tracking_implementation_sha256": document_digest(
            source_digests(ROOT / "src/tennis_scene/multiview_tracking")
        ),
        "evaluation_sha256": {
            filename: dual_sha256(RUN / filename)
            for filename in ("measure.py", "evaluate_projection.py")
        },
    }


def holdout(plan_path: Path, poses: Path, output: Path) -> dict[str, Any]:
    attempt = ROOT / "outputs/i2445/holdout/scoring-attempt.json"
    if attempt.exists() or output.exists():
        raise FileExistsError(
            "The single reserved tracking scoring attempt already started"
        )
    protocol = reserved_protocol()
    if protocol != read_json(INTEGRATION_EVIDENCE / "reserved-protocol.json"):
        raise ValueError(
            "The tracking solver or scorer differs from the frozen holdout protocol"
        )
    prep = read_json(C_EVIDENCE / "holdout-pose-preparation.json")
    plan_sha = dual_sha256(plan_path)
    plan = read_json(plan_path)
    if (
        plan_sha != prep["plan_sha256"]
        or plan["solver_commit"] != prep["solver_commit"]
        or plan["solver_config"] != json_value(DEFAULT_CONFIG)
    ):
        raise ValueError("The reserved ViTPose plan differs from the pinned #2444 plan")
    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            plan["solver_commit"],
            "--",
            "src/tennis_scene/multiview_pose/*.py",
        ],
        cwd=ROOT,
        check=True,
    )
    complete_path = poses / "complete.json"
    if not complete_path.is_file():
        raise FileNotFoundError(
            f"All nine reserved camera pose ports are required: {complete_path}"
        )
    complete = read_json(complete_path)
    entries = plan["clips"]
    if (
        complete["plan_sha256"] != plan_sha
        or complete["pose_model"] != plan["pose_model"]
        or set(entries) != set(complete["clips"])
    ):
        raise ValueError("Expected completed ViTPose for all three reserved clips")
    repair_path = C_EVIDENCE / "holdout-camera-recovery.json"
    repair = read_json(repair_path)
    pins = _label_pins(Path(plan["dataset"]), set(entries))
    evidence = {
        **protocol,
        "plan_sha256": plan_sha,
        "completed_pose_sha256": dual_sha256(complete_path),
        "camera_recovery_sha256": dual_sha256(repair_path),
        "labels_sha256": {clip: sha for clip, (_, sha) in pins.items()},
    }
    prepared = {}
    for clip, entry in entries.items():
        inputs = _observations(
            poses / "stores",
            clip,
            entry,
            complete["clips"][clip],
            plan_sha,
            Path(plan["archive"]),
            repair,
        )
        candidates = reconstruct(inputs)
        start = perf_counter()
        result = track(candidates, fps=inputs.source.fps, config=SINGLES)
        prepared[clip] = (inputs, result, perf_counter() - start)

    attempt.parent.mkdir(parents=True, exist_ok=True)
    with attempt.open("x", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)
    clips, view_sets = {}, {}
    for clip, (inputs, result, seconds) in prepared.items():
        label, sha = pins[clip]
        labels = ClipLabels.load(label)
        clips[clip], view_sets[clip] = _score_clip(
            inputs, labels, result, {"labels_sha256": sha}, seconds
        )
    return {
        "pins": evidence,
        "clips": clips,
        "all_three": _summarize(view_sets, sorted(clips)),
        "without_repaired_clip": _summarize(
            view_sets, sorted(clips.keys() - {REPAIRED_CLIP})
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    dev = modes.add_parser("development")
    dev.add_argument("--dataset", type=Path, required=True)
    dev.add_argument("--archive", type=Path, required=True)
    dev.add_argument("--output", type=Path, required=True)
    four = modes.add_parser("doubles")
    four.add_argument("--candidate-stores", type=Path, required=True)
    four.add_argument("--output", type=Path, required=True)
    reserved = modes.add_parser("holdout")
    reserved.add_argument("--plan", type=Path, required=True)
    reserved.add_argument("--poses", type=Path, required=True)
    reserved.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    output = args.output
    if args.mode == "development":
        metrics = development(args.dataset, args.archive)
    elif args.mode == "doubles":
        metrics = doubles(args.candidate_stores, output)
        output = output / "metrics.json"
    else:
        metrics = holdout(args.plan, args.poses, output)
    metrics["wall_seconds"] = perf_counter() - started
    metrics["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json_atomic(output, metrics)


if __name__ == "__main__":
    main()
