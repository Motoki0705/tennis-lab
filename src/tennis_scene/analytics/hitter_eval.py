"""Render prediction-blind multiview evidence and evaluate the V4 hitter rule."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.tennis_scene.analytics.hitter_v4 import SPLITS, V4Benchmark, V4Clip, V4Shot

if TYPE_CHECKING:
    from src.tennis_scene.analytics.hitter import HitterScene, ShotHitter

EVALUATION_SPLITS = ("val", "test")
EXPECTED_CLIP_COUNTS = {"train": 47, "val": 11, "test": 11}
VISIBILITY_STATUSES = {"confirmed", "confirmed_with_track_gap", "unidentifiable"}


@dataclass(frozen=True, slots=True)
class VisualLabel:
    clip_id: str
    source_event_id: str
    frame: int
    subtype: str
    split: str
    gt_local_player_id: int | None
    gt_local_player_index: int | None
    visibility_status: str
    audit_note: str

    @property
    def key(self) -> tuple[str, str]:
        return self.clip_id, self.source_event_id


def read_labels(
    path: Path, benchmark: V4Benchmark
) -> tuple[dict[tuple[str, str], VisualLabel], str]:
    data = path.read_bytes()
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
    labels = {}
    for row in reader:
        label = VisualLabel(
            row["clip_id"],
            row["source_event_id"],
            int(row["frame"]),
            row["subtype"],
            row["split"],
            int(row["gt_local_player_id"]) if row["gt_local_player_id"] else None,
            int(row["gt_local_player_index"]) if row["gt_local_player_index"] else None,
            row["visibility_status"],
            row["audit_note"],
        )
        clip = benchmark.clips.get(label.clip_id)
        known = label.visibility_status != "unidentifiable"
        if (
            clip is None
            or label.split != clip.split
            or label.split not in EVALUATION_SPLITS
            or label.visibility_status not in VISIBILITY_STATUSES
            or not label.audit_note.strip()
            or (label.gt_local_player_id is not None) != known
            or (label.gt_local_player_index is not None) != known
            or label.key in labels
        ):
            raise ValueError(f"Invalid or duplicate ground truth for {label.key}")
        labels[label.key] = label
    return labels, sha256(data).hexdigest()


def labeled_shots(
    benchmark: V4Benchmark,
    labels: dict[tuple[str, str], VisualLabel],
    split: str,
) -> Iterable[tuple[VisualLabel, V4Shot, HitterScene]]:
    from src.tennis_scene.analytics.hitter import HitterScene

    remaining = {key for key, label in labels.items() if label.split == split}
    for clip in benchmark.clips.values():
        if clip.split != split:
            continue
        with np.load(clip.scene.verify(), allow_pickle=False) as archive:
            shots = clip.shots(archive)
            scene = HitterScene.from_archive(archive)
        if len(scene.ball) != clip.frames or scene.players.track_ids is None:
            raise ValueError(f"{clip.clip_id}: invalid player IDs or ball timeline")
        tracks = scene.players.track_ids.tolist()
        for shot in shots:
            key = (clip.clip_id, shot.source_event_id)
            label = labels.get(key)
            if label is None or label.split != split:
                raise ValueError(f"Missing visual label for V4 shot {key}")
            if (label.frame, label.subtype) != (shot.frame, shot.subtype):
                raise ValueError(f"Visual label disagrees with V4 shot {key}")
            index = label.gt_local_player_index
            if index is not None and (
                not 0 <= index < len(tracks)
                or tracks[index] != label.gt_local_player_id
            ):
                raise ValueError(
                    f"Visual label has an invalid local player ID/index for {key}"
                )
            remaining.remove(key)
            yield label, shot, scene
    if remaining:
        raise ValueError(f"Visual labels have no matching shot: {sorted(remaining)}")


@dataclass(frozen=True, slots=True)
class _SourceRange:
    clip_id: str
    split: str
    start: float
    end: float


def _overlaps(
    ranges: Iterable[_SourceRange], *, inclusive: bool = False
) -> list[dict[str, Any]]:
    overlaps = []
    for first, second in combinations(ranges, 2):
        if first.split == second.split:
            continue
        start = max(first.start, second.start)
        end = min(first.end, second.end)
        if start < end or (inclusive and start == end):
            overlaps.append(
                {
                    "clip_ids": [first.clip_id, second.clip_id],
                    "splits": [first.split, second.split],
                    "overlap": [start, end],
                }
            )
    return overlaps


def _source_ranges(
    clip: V4Clip, source: dict[str, Any]
) -> tuple[str, _SourceRange, list[tuple[str, _SourceRange]]]:
    source_id = f"{source['dataset_id']}/{source['video_id']}"
    if source["clip_id"] != clip.clip_id or source["num_frames"] != clip.frames:
        raise ValueError(f"{clip.clip_id}: invalid V4 source timeline")
    global_start = source["global_start_sec"]
    global_end = source["global_end_sec"]
    if not 0 <= global_start < global_end:
        raise ValueError(f"{clip.clip_id}: invalid V4 source time range")
    cameras = []
    for camera in source["cameras"]:
        camera_id = camera["camera_id"]
        start, end = camera["source_frame_start"], camera["source_frame_end"]
        if not 0 <= start <= end:
            raise ValueError(f"{clip.clip_id}/{camera_id}: invalid source frame range")
        raw_name = Path(camera["source_path"]).name
        cameras.append(
            (
                f"{source_id}/{raw_name}",
                _SourceRange(clip.clip_id, clip.split, start, end),
            )
        )
    if {camera["camera_id"] for camera in source["cameras"]} != clip.cameras.keys():
        raise ValueError(f"{clip.clip_id}: V4 source camera mismatch")
    return (
        source_id,
        _SourceRange(clip.clip_id, clip.split, global_start, global_end),
        cameras,
    )


def audit_splits(benchmark: V4Benchmark) -> dict[str, Any]:
    clips = Counter(clip.split for clip in benchmark.clips.values())
    if clips != EXPECTED_CLIP_COUNTS:
        raise ValueError(f"Expected the fixed V4 47/11/11 split, got {dict(clips)}")
    videos: dict[str, list[_SourceRange]] = defaultdict(list)
    camera_ranges: dict[str, list[_SourceRange]] = defaultdict(list)
    media_by_hash: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    shot_counts: Counter[str] = Counter()
    for clip in benchmark.clips.values():
        source_id, timeline, cameras = _source_ranges(clip, clip.clip_json.read_json())
        videos[source_id].append(timeline)
        for name, coverage in cameras:
            camera_ranges[name].append(coverage)
        for camera_id, ref in clip.cameras.items():
            media_by_hash[ref.sha256].append((clip.clip_id, camera_id, clip.split))
        shot_counts[clip.split] += clip.shot_count
    return {
        "clip_counts": {split: clips[split] for split in SPLITS},
        "shot_counts": {split: shot_counts[split] for split in SPLITS},
        "duplicate_clip_ids_across_splits": [],
        "shared_source_videos": {
            name: sorted(splits)
            for name, ranges in videos.items()
            if len(splits := {clip.split for clip in ranges}) > 1
        },
        "cross_split_time_overlaps_sec": [
            {"source_video": name, **pair}
            for name, ranges in videos.items()
            for pair in _overlaps(ranges)
        ],
        "cross_split_source_frame_overlaps_inclusive": [
            {"source_camera": name, **pair}
            for name, ranges in camera_ranges.items()
            for pair in _overlaps(ranges, inclusive=True)
        ],
        "cross_split_duplicate_media_sha256": {
            digest: [list(item) for item in entries]
            for digest, entries in media_by_hash.items()
            if len({entry[2] for entry in entries}) > 1
        },
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _error_evidence(label: VisualLabel, estimate: ShotHitter) -> dict[str, Any]:
    truth = next(
        candidate
        for candidate in estimate.candidates
        if candidate.local_player_id == label.gt_local_player_id
    )
    winner = estimate.candidates[0]
    if truth.contact is None:
        reason = "gt_without_jointly_valid_ball_landmark"
    elif truth.contact.relative_distance > 1:
        reason = "gt_landmark_outside_reach"
    elif truth.contact.side_alignment in ("mismatch", "unknown"):
        reason = "gt_court_side_disagrees_or_missing"
    else:
        reason = "other_player_ranked_higher"
    return {
        **asdict(label),
        "estimated_local_player_id": estimate.local_player_id,
        "margin_to_second": estimate.margin_to_second,
        "reason": reason,
        "estimated_candidate": asdict(winner),
        "gt_candidate": asdict(truth),
    }


def summarize(
    results: Iterable[tuple[VisualLabel, ShotHitter]],
) -> dict[str, Any]:
    statuses: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    visibility: Counter[str] = Counter()
    evaluable = evaluable_resolved = correct = 0
    errors = []
    for label, estimate in results:
        status = estimate.resolution_status
        statuses[status] += 1
        visibility[label.visibility_status] += 1
        if estimate.rejection_reason:
            reasons[estimate.rejection_reason] += 1
        if label.gt_local_player_id is None:
            continue
        evaluable += 1
        if status == "resolved":
            evaluable_resolved += 1
            if estimate.local_player_id == label.gt_local_player_id:
                correct += 1
            else:
                errors.append(_error_evidence(label, estimate))
    total = statuses.total()
    return {
        "total_shots": total,
        "evaluable_shots": evaluable,
        "resolved_count": statuses["resolved"],
        "ambiguous_count": statuses["ambiguous"],
        "unresolved_count": statuses["unresolved"],
        "evaluable_resolved_count": evaluable_resolved,
        "correct_resolved_count": correct,
        "false_resolution_count": len(errors),
        "coverage": _ratio(statuses["resolved"], total),
        "resolved_accuracy": _ratio(correct, evaluable_resolved),
        "false_resolution_rate": _ratio(len(errors), evaluable),
        "false_resolution_rate_among_evaluable_resolved": _ratio(
            len(errors), evaluable_resolved
        ),
        "overall_accuracy": _ratio(correct, evaluable),
        "rejection_reasons": dict(sorted(reasons.items())),
        "visibility_statuses": dict(sorted(visibility.items())),
        "false_resolutions": errors,
    }


def evaluate(benchmark: V4Benchmark, gt: Path, split: str) -> dict[str, Any]:
    from src.tennis_scene.analytics.hitter import DEFAULT_THRESHOLDS, estimate_hitter

    split_audit = audit_splits(benchmark)
    labels, gt_sha = read_labels(gt, benchmark)
    results = (
        (
            label,
            estimate_hitter(
                scene,
                frame=shot.frame,
                frame_uncertainty=shot.frame_uncertainty,
                court_side=shot.court_side,
            ),
        )
        for label, shot, scene in labeled_shots(benchmark, labels, split)
    )
    return {
        "schema": "v4_hitter_evaluation.v1",
        "benchmark_id": benchmark.benchmark_id,
        "manifest_sha256": benchmark.manifest_sha256,
        "ground_truth_sha256": gt_sha,
        "rule_version": DEFAULT_THRESHOLDS.rule_version,
        "thresholds": asdict(DEFAULT_THRESHOLDS),
        "split": split,
        "split_audit": split_audit,
        "metrics": summarize(results),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    render = commands.add_parser("render", help="render shot panels for visual review")
    render.add_argument("--output", required=True, type=Path)
    render.add_argument("--clip-id", action="append")
    score = commands.add_parser(
        "evaluate", help="evaluate the frozen rule on one split"
    )
    score.add_argument("--ground-truth", required=True, type=Path)
    score.add_argument("--split", choices=EVALUATION_SPLITS, required=True)
    args = parser.parse_args(argv)
    try:
        benchmark = V4Benchmark.load(args.benchmark, args.manifest_sha256)
        if args.command == "render":
            from src.tennis_scene.analytics.hitter_audit import render_audit

            if args.output.resolve().is_relative_to(args.benchmark.resolve()):
                raise ValueError("Audit output must be outside the frozen benchmark")
            clip_ids = args.clip_id or [
                clip.clip_id
                for clip in benchmark.clips.values()
                if clip.split in EVALUATION_SPLITS
            ]
            render_audit(benchmark, args.output, clip_ids)
        else:
            print(
                json.dumps(
                    evaluate(benchmark, args.ground_truth, args.split),
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"hitter_eval: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
