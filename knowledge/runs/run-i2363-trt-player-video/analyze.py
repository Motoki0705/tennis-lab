"""Summarize the completed #2363 video run from its progress, trace and artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import mmap
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.tennis_scene.motion_alignment.temporal import PlacementRejection
from src.tennis_scene.pipeline.storage.scene_index import annotation_scene_path
from src.tennis_scene.schema import SCENE_MASK_FIELDS
from src.tennis_scene.visualization.review.store import StoreReader
from src.utils.geometry.triangulation import PointRejection
from src.utils.io import write_json_atomic

SAMPLES_START = b',"resource_samples":['
TIMINGS_START = b'],"timings":'
EVENTS_START = b',"events":['
EVENT_PREFIX = b'{"id":'
SAMPLE_DURATION = re.compile(rb'"sample_duration_ns":(\d+)')
JOINT_REJECTION_NAMES = {int(reason): reason.name for reason in PointRejection}
ROOT_REJECTION_NAMES = JOINT_REJECTION_NAMES | {
    int(reason): reason.name for reason in PlacementRejection
}


def recording_header(path: Path) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Read aggregates and sample costs without materializing the two large arrays."""
    with (
        path.open("rb") as stream,
        mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as trace,
    ):
        samples = trace.find(SAMPLES_START)
        timings = trace.find(TIMINGS_START, samples + len(SAMPLES_START))
        events = trace.find(EVENTS_START, timings + len(TIMINGS_START))
        if not 0 < samples < timings < events or trace[-3:] != b"]}\n":
            raise ValueError(f"Incomplete or unsupported recording: {path}")
        result: dict[str, Any] = json.loads(
            trace[:samples] + trace[timings + 1 : events] + b"}"
        )
        durations = [
            int(match[1]) for match in SAMPLE_DURATION.finditer(trace, samples, timings)
        ]
    if (
        result["schema"] != "inference_recording_v1"
        or result["status"] != "complete"
        or result["measurement_errors"]
    ):
        raise ValueError(f"Inference recording did not complete cleanly: {path}")
    sample_stats = (
        {
            "count": len(durations),
            "duration_median_ms": float(np.median(durations)) / 1e6,
        }
        if durations
        else {"count": 0}
    )
    return result, sample_stats, events + len(EVENTS_START)


def transfer_operations(path: Path, offset: int) -> list[dict[str, Any]]:
    """Sum recorded host spans by transfer operation, reading the event array in chunks."""
    times: Counter[tuple[str, str]] = Counter()
    cpu: Counter[tuple[str, str]] = Counter()
    count: Counter[tuple[str, str]] = Counter()

    def record(raw: bytes) -> None:
        if b',"stage":"transfer",' in raw:
            event = json.loads(EVENT_PREFIX + raw)
            attrs = event["attributes"]
            key = attrs.get("model", ""), attrs.get("operation", "")
            times[key] += event["end_ns"] - event["start_ns"]
            cpu[key] += event["thread_cpu_ns"]
            count[key] += 1

    with path.open("rb") as stream:
        stream.seek(offset)
        prefix = stream.read(len(EVENT_PREFIX))
        if prefix == b"]}\n":
            return []
        if prefix != EVENT_PREFIX:
            raise ValueError(f"Expected events written in event id order: {path}")
        tail = b""
        while block := stream.read(4 * 1024 * 1024):
            chunks = (tail + block).split(b"}," + EVENT_PREFIX)
            tail = chunks.pop()
            for raw in chunks:
                record(raw + b"}")
        record(tail[:-3])
    return [
        {
            "model": key[0],
            "operation": key[1],
            "spans": count[key],
            "wall_seconds": elapsed / 1e9,
            "thread_cpu_seconds": cpu[key] / 1e9,
        }
        for key, elapsed in times.most_common()
    ]


def performance_summary(
    record: dict[str, Any], outcome: dict[str, Any]
) -> dict[str, Any]:
    timing = record["timings"]
    by_operation: dict[tuple[str, str, str], list[float]] = defaultdict(
        lambda: [0.0, 0.0]
    )
    for item in timing["cpu_during_gpu_idle"]:
        key = item["stage"], item["operation"], item["model"]
        by_operation[key][0] += item["estimated_thread_cpu_seconds"]
        by_operation[key][1] += item["thread_wall_seconds"]
    aggregate_idle = [
        {
            "stage": stage,
            "operation": operation,
            "model": model,
            "estimated_thread_cpu_seconds": cpu,
            "thread_wall_seconds": wall,
        }
        for (stage, operation, model), (cpu, wall) in sorted(
            by_operation.items(), key=lambda row: (-row[1][0], row[0])
        )
    ]
    return {
        "cli_wall_seconds": outcome["wall_seconds"],
        "cli_exit_code": outcome["exit_code"],
        "recording_wall_seconds": record["total_wall_seconds"],
        "clock": record["timing"],
        "resources": record["resources"],
        "working_memory": record["metadata"]["budget"],
        "exclusive_stage_wall_seconds": record["stage_wall_seconds"],
        "batch_dispatches": record["batch_dispatches"],
        "model_lifecycle": record["models"],
        "by_stage": timing["by_stage"],
        "by_node": timing["by_node"],
        "by_model": timing["by_model"],
        "cpu_during_gpu_idle_by_operation": aggregate_idle,
        "cpu_during_gpu_idle_by_node": timing["cpu_during_gpu_idle"],
    }


def progress_path(run: Path) -> Path:
    (progress,) = (run / "all/progress").glob("*/progress.json")
    return progress


def clip_store(run: Path, record: dict[str, Any]) -> StoreReader:
    path = (
        run / "inputs/dataset" / record["path"] / "annotations/tennis_scene/scene.json"
    )
    store = StoreReader(path)
    if (
        store.frame_count != record["num_frames"]
        or len(store.camera_ids) != record["num_cameras"]
    ):
        raise ValueError(f"Scene store does not match the target manifest: {path}")
    return store


def tracking_rows(store: StoreReader, clip_id: str) -> list[dict[str, Any]]:
    rows = []
    for camera in store.camera_ids:
        tracks, _ = store.payload(f"person_tracking/{camera}")
        selection, _ = store.payload(f"player_selection/{camera}")
        linking = selection["diagnostics"]["linking"]
        groups = selection["tracks"]
        rows.append(
            {
                "clip_id": clip_id,
                "camera": camera,
                "stable_track_ids": tracks["track_ids"].tolist(),
                "tracking_observations": int(tracks["observed"].sum()),
                "source_tracklets": sum(map(len, tracks["source_track_ids"])),
                "tracklet_links": len(tracks["tracklet_links"]),
                "candidate_groups": len(linking["groups"]),
                "selected_group_ids": groups["track_ids"].tolist(),
                "selected_group_frames": groups["observed"].sum(axis=1).tolist(),
                "retained_raw_observations": int(selection["selected"].sum()),
                "required_frames": linking["required_frames"],
                "selected_invalid_footpoints": linking["selected_invalid_footpoints"],
                "selected_outside_corridor": linking["selected_outside_corridor"],
            }
        )
    return rows


def association_row(store: StoreReader, outcome: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    diagnostics = outcome.get("diagnostics", {})
    if "court_side" in store.references:
        side, _ = store.payload("court_side")
    else:
        side = diagnostics
    if hypotheses := side.get("hypotheses"):
        output.update(
            court_side_frames=side["frames"],
            court_side_margin=side["margin"],
            court_side_cost=hypotheses[0]["cost"],
            court_side_support=hypotheses[0]["support"],
        )
    if "player_association" in store.references:
        association, _ = store.payload("player_association")
        diagnostics = association["diagnostics"]
        assignment = association["player_ids"]
        views = np.stack(
            [(assignment == player).any(axis=1).sum(axis=0) for player in range(4)]
        )
        output["assigned_view_counts_by_player"] = [
            np.bincount(counts, minlength=len(store.camera_ids) + 1).tolist()
            for counts in views
        ]
    if evidence := diagnostics.get("association"):
        output.update(
            person_candidates=len(evidence["candidates"]),
            identity_presence_frames={
                i["identity"]: [i["side"], i["presence_frames"]]
                for i in evidence["identities"]
            },
            global_player_selection=evidence["selection"],
            min_player_margin=evidence.get("min_player_margin"),
            ambiguous_pairs=len(evidence["ambiguous_pairs"])
            if "ambiguous_pairs" in evidence
            else None,
            undecided_candidates=len(evidence["undecided_candidates"])
            if "undecided_candidates" in evidence
            else None,
            handoff_frames=sum(item["frames"] for item in evidence["handoff_frames"])
            if "handoff_frames" in evidence
            else None,
        )
    return output


def masks_row(store: StoreReader, target: dict[str, Any]) -> dict[str, Any]:
    annotation = json.loads((store.root / "annotation.json").read_bytes())
    if (
        annotation["clip_manifest_sha256"] != target["manifest_sha256"]
        or annotation["media_sha256"] != target["media_sha256"]
    ):
        raise ValueError(
            f"Published input hashes differ from run targets: {store.root}"
        )
    result: dict[str, Any] = {
        "result_status": annotation["result_status"],
        "scene_archive": annotation["scene_result"],
    }
    with np.load(
        annotation_scene_path(store.root, annotation), allow_pickle=False
    ) as archive:
        for name in SCENE_MASK_FIELDS:
            mask = archive[name]
            expected: tuple[int, ...] = (
                (store.frame_count,)
                if name == "ball_3d_valid"
                else (4, store.frame_count)
            )
            if name == "player_kp_3d_vis":
                expected += (17,)
            if mask.shape != expected or mask.dtype != np.bool_:
                raise ValueError(f"Unexpected mask {name}: {store.root}")
            valid = int(mask.sum())
            result.update(
                {
                    f"{name}_count": valid,
                    f"{name}_total": mask.size,
                    f"{name}_fraction": valid / mask.size,
                }
            )
        joint_frames = archive["player_kp_3d_vis"].any(axis=-1).sum(axis=1).tolist()
        root_frames = archive["player_valid"].sum(axis=1).tolist()
        smpl_frames = archive["player_smpl_valid"].sum(axis=1).tolist()
        stats = annotation["validity_statistics"]
        if [joint_frames, root_frames, smpl_frames, result["ball_3d_valid_count"]] != [
            stats["player_joint_frames"],
            stats["player_root_frames"],
            stats["player_smpl_frames"],
            0,
        ] or stats["ball_3d_frames"] != 0:
            raise ValueError(
                f"Published validity counts or disabled ball disagree with masks: {store.root}"
            )
        result.update(
            joint_frames_by_player=joint_frames,
            root_frames_by_player=root_frames,
            smpl_frames_by_player=smpl_frames,
        )
        for field, names in (
            ("player_rejection_code", ROOT_REJECTION_NAMES),
            ("player_kp_3d_rejection_code", JOINT_REJECTION_NAMES),
        ):
            codes, counts = np.unique(archive[field], return_counts=True)
            result[field] = {
                names[int(code)]: int(count)
                for code, count in zip(codes, counts, strict=True)
            }
    return result


def quality_rows(
    run: Path, progress: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clips, cameras = [], []
    targets = {item["record"]["clip_id"]: item for item in progress["targets"]}
    if targets.keys() != progress["clips"].keys():
        raise ValueError("Video progress does not contain an outcome for each target")
    for clip_id, target in sorted(targets.items()):
        record = target["record"]
        outcome = progress["clips"][clip_id]
        nodes = progress["nodes"][clip_id]
        store = clip_store(run, record)
        row = {
            "clip_id": clip_id,
            "frames": record["num_frames"],
            "status": outcome["status"],
            "failed_nodes": [
                node for node, value in nodes.items() if value["status"] == "failed"
            ],
            "blocked_nodes": sum(
                value["status"] == "blocked" for value in nodes.values()
            ),
            "reason": outcome.get("reason"),
            "error": outcome["error"],
        }
        row.update(association_row(store, outcome))
        if outcome["status"] == "generated":
            row.update(masks_row(store, target))
        elif (store.root / "annotation.json").exists() or store.document.get(
            "exports", {}
        ).get("scene") is not None:
            raise ValueError(f"Unsuccessful clip has a published scene: {clip_id}")
        clips.append(row)
        cameras.extend(tracking_rows(store, clip_id))
    return clips, cameras


def write_csv(destination: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    path = progress_path(args.run)
    progress = json.loads(path.read_bytes())
    outcome = json.loads((args.run / "all/outcome.json").read_bytes())
    clips, cameras = quality_rows(args.run, progress)
    if {row["clip_id"]: row["status"] for row in clips} != outcome["statuses"]:
        raise ValueError("The progress and CLI outcomes disagree")
    trace_path = path.parent / progress["recording"]
    record, sample_stats, offset = recording_header(trace_path)
    transfers = transfer_operations(trace_path, offset)
    if not math.isclose(
        sum(row["wall_seconds"] for row in transfers),
        record["stage_inclusive_seconds"]["transfer"],
    ):
        raise ValueError("Transfer events do not cover the recorded transfer time")
    performance = performance_summary(record, outcome)
    performance.update(
        sample_statistics=sample_stats,
        transfer_operations=transfers,
        trace_bytes=trace_path.stat().st_size,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "clips.csv", clips)
    write_csv(args.output / "camera-tracks.csv", cameras)
    write_json_atomic(args.output / "performance.json", performance)
    print(dict(Counter(row["status"] for row in clips)))


if __name__ == "__main__":
    main()
