"""Compare frozen Refiner and rule events against independent RGB labels."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from collections.abc import Sequence
from itertools import groupby
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
from protocol import (
    BENCHMARK_ID,
    CLASSES,
    METHODS,
    PINS,
    PRIMARY_TOLERANCE,
    TOLERANCES,
    load_labels,
    read_json,
    save_json,
    sha256,
)
from scipy.optimize import linear_sum_assignment

Row = dict[str, Any]


def pair_frames(
    left: Sequence[int], right: Sequence[int], tolerance: int
) -> list[tuple[int, int]]:
    """Maximize the number of pairs, then minimize their total absolute frame error."""
    if not left or not right:
        return []
    distances = np.abs(np.asarray(left)[:, None] - np.asarray(right)[None, :])
    forbidden = (min(len(left), len(right)) + 1) * (tolerance + 1)
    rows, columns = linear_sum_assignment(
        np.where(distances <= tolerance, distances, forbidden)
    )
    return [
        (int(row), int(col))
        for row, col in zip(rows, columns, strict=True)
        if distances[row, col] <= tolerance
    ]


def load_prediction(path: Path, clip: Row, progress: Row, method: str) -> Row:
    key = clip["clip_id"]
    prediction = read_json(path, progress.get("sha256"))
    version = PINS["checkpoint_sha256"] if method == "refiner3" else PINS["rule_commit"]
    status = progress["status"]
    expected = {
        "schema": "ball_event_predictions.v1",
        "clip_id": key,
        "fps": clip["fps"],
        "frames": clip["frames"],
        "method": method,
        "method_version": version,
        "status": status,
        "reason": progress["reason"],
    }
    if status == "ok":
        expected["inputs"] = {"scene_sha256": clip["scene"]["sha256"]}
    if status not in {"ok", "failed"} or any(
        prediction.get(field) != value for field, value in expected.items()
    ):
        raise ValueError(
            f"Prediction provenance or frame grid mismatch: {method} {key}"
        )
    events = prediction["events"]
    if status == "failed" and (events or not prediction["reason"]):
        raise ValueError(f"Invalid failed prediction: {method} {key}")
    counts: Counter[str] = Counter()
    seen = set()
    for event in events:
        kind, frame = event["kind"], event["frame"]
        if (
            (kind, frame) in seen
            or kind not in CLASSES
            or type(frame) is not int
            or not 0 <= frame < clip["frames"]
            or not math.isclose(
                event["time_s"], frame / clip["fps"], rel_tol=0, abs_tol=1e-9
            )
            or not 0 <= event["score"] <= 1
            or not isinstance(event["features"], dict)
            or not isinstance(event["flags"], list)
        ):
            raise ValueError(f"Invalid event: {method} {key} {event}")
        counts[kind] += 1
        seen.add((kind, frame))
    if any(progress["events"].get(kind, 0) != counts[kind] for kind in CLASSES):
        raise ValueError(f"Progress counts do not match predictions: {method} {key}")
    return prediction


def label_mask(label: Row) -> set[int]:
    return {
        frame
        for interval in label["unlabelable"]
        for frame in range(interval["start"], interval["end"] + 1)
    }


def score_events(
    label_events: list[Row], predictions: list[Row], masked: set[int], tolerance: int
) -> Row:
    targets = sorted(label_events, key=lambda event: (event["frame"], event["id"]))
    candidates = sorted(predictions, key=lambda event: event["frame"])
    eligible = [
        x for x in targets if x["frame"] not in masked and x["status"] != "uncertain"
    ]
    uncertain = [
        x for x in targets if x["frame"] not in masked and x["status"] == "uncertain"
    ]
    available = [x for x in candidates if x["frame"] not in masked]
    pairs = pair_frames(
        [x["frame"] for x in available], [x["frame"] for x in eligible], tolerance
    )
    used_predictions, used_labels = {i for i, _ in pairs}, {j for _, j in pairs}
    remaining = [x for i, x in enumerate(available) if i not in used_predictions]
    ignored_pairs = pair_frames(
        [x["frame"] for x in remaining], [x["frame"] for x in uncertain], tolerance
    )
    ignored_indices = {i for i, _ in ignored_pairs}
    return {
        "matches": [
            {
                "prediction": available[i],
                "label": eligible[j],
                "offset_frames": available[i]["frame"] - eligible[j]["frame"],
            }
            for i, j in pairs
        ],
        "false_positives": [
            x for i, x in enumerate(remaining) if i not in ignored_indices
        ],
        "false_negatives": [x for j, x in enumerate(eligible) if j not in used_labels],
        "uncertain_labels": [x for x in targets if x["status"] == "uncertain"],
        "masked_labels": [
            x for x in targets if x["frame"] in masked and x["status"] != "uncertain"
        ],
        "masked_predictions": [x for x in candidates if x["frame"] in masked],
        "uncertain_predictions": [
            {"prediction": remaining[i], "label": uncertain[j]}
            for i, j in ignored_pairs
        ],
    }


def report(rows: list[Row], fps: float) -> Row:
    offsets = [pair["offset_frames"] for row in rows for pair in row["matches"]]
    matched_by_point = Counter(
        pair["label"]["in_point"] for row in rows for pair in row["matches"]
    )
    missed_by_point = Counter(
        label["in_point"] for row in rows for label in row["false_negatives"]
    )
    tp = len(offsets)
    fp = sum(len(row["false_positives"]) for row in rows)
    fn = sum(len(row["false_negatives"]) for row in rows)
    absolute = [abs(x) for x in offsets]
    med = median(offsets) if offsets else None
    med_abs = median(absolute) if offsets else None
    mae = mean(absolute) if offsets else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "labels": tp + fn,
        "predictions": tp + fp,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
        "median_signed_error_frames": med,
        "median_abs_error_frames": med_abs,
        "mean_abs_error_frames": mae,
        "median_signed_error_ms": med * 1000 / fps if med is not None else None,
        "median_abs_error_ms": med_abs * 1000 / fps if med_abs is not None else None,
        "mean_abs_error_ms": mae * 1000 / fps if mae is not None else None,
        "by_point_context": {
            name: {
                "matched": matched_by_point[inside],
                "missed": missed_by_point[inside],
            }
            for name, inside in (("in_point", True), ("out_of_point", False))
        },
        "excluded": {
            key: sum(len(row[key]) for row in rows)
            for key in (
                "uncertain_labels",
                "masked_labels",
                "masked_predictions",
                "uncertain_predictions",
            )
        },
    }


def point_spans(label: Row) -> list[tuple[int, int]]:
    events = sorted(label["events"], key=lambda event: event["frame"])
    spans = []
    for inside, group in groupby(events, key=lambda event: event["in_point"]):
        frames = [event["frame"] for event in group]
        if inside:
            spans.append((frames[0], frames[-1]))
    return spans


def ball_context(benchmark: Path, clip: Row) -> tuple[np.ndarray, np.ndarray]:
    path = benchmark / clip["scene"]["path"]
    scene = read_json(path, clip["scene"]["sha256"])
    archive = path.parent / scene["exports"]["scene"]["files"]["scene"]["path"]
    with np.load(archive, allow_pickle=False) as arrays:
        return arrays["ball_3d"], arrays["ball_3d_valid"]


def error_tags(
    row: Row, label: Row, preds: list[Row], ball: np.ndarray, valid: np.ndarray
) -> Row:
    """Record independent context about mismatches, without assigning a cause."""
    spans = point_spans(label)
    counts: dict[str, Counter[str]] = {"fp": Counter(), "fn": Counter()}
    detail = {"fp": [], "fn": []}
    for side, events, opposite in (
        ("fp", row["false_positives"], label["events"]),
        ("fn", row["false_negatives"], preds),
    ):
        for event in events:
            frame = event["frame"]
            neighborhood = valid[max(frame - 5, 0) : frame + 6]
            tags = []
            if side == "fn" and not event["in_point"]:
                tags.append("out_of_point_label")
            if side == "fp" and not any(
                start - 5 <= frame <= stop + 5 for start, stop in spans
            ):
                tags.append("outside_point_spans")
            if not neighborhood.all():
                tags.append("3d_missing_within_5_frames")
            position = [float(v) for v in ball[frame]] if valid[frame] else None
            if position is not None and abs(position[1]) <= 1.0:
                tags.append("3d_within_1m_of_net_plane")
            if side == "fn" and event.get("subtype") == "volley":
                tags.append("volley_label")
            if side == "fn" and any(
                word in event["evidence"] for word in ("隠れ", "遮蔽")
            ):
                tags.append("occlusion_in_label_evidence")
            nearby = sorted(
                [other for other in opposite if abs(other["frame"] - frame) <= 10],
                key=lambda other: abs(other["frame"] - frame),
            )
            if any(
                other["kind"] == event["kind"] and abs(other["frame"] - frame) > 5
                for other in nearby
            ):
                tags.append("same_kind_6_to_10_frames_apart")
            if any(
                other["kind"] != event["kind"] and abs(other["frame"] - frame) <= 5
                for other in nearby
            ):
                tags.append("different_kind_within_5_frames")
            counts[side].update(tags)
            detail[side].append(
                {
                    "event": event,
                    "contexts": tags,
                    "ball_3d_m": position,
                    "nearby_other_events": [
                        {
                            k: v
                            for k, v in other.items()
                            if k in {"kind", "frame", "status", "score"}
                        }
                        for other in nearby
                    ],
                }
            )
    return {"counts": counts, "details": detail}


def distribution(counts: list[int]) -> Row:
    return {
        "clips": len(counts),
        "events": sum(counts),
        "min": min(counts) if counts else None,
        "median": median(counts) if counts else None,
        "p90": float(np.percentile(counts, 90)) if counts else None,
        "max": max(counts) if counts else None,
        "histogram": dict(
            sorted(Counter(map(str, counts)).items(), key=lambda pair: int(pair[0]))
        ),
    }


def agreement(keys: list[str], predictions: dict[str, dict[str, Row]]) -> Row:
    paired = [
        key
        for key in keys
        if all(
            predictions[method].get(key, {}).get("status") == "ok" for method in METHODS
        )
    ]
    per_clip = {}
    for key in paired:
        per_clip[key] = {}
        for kind in CLASSES:
            frames = [
                [
                    event["frame"]
                    for event in predictions[method][key]["events"]
                    if event["kind"] == kind
                ]
                for method in METHODS
            ]
            per_clip[key][kind] = {
                **{
                    method: len(values)
                    for method, values in zip(METHODS, frames, strict=True)
                },
                "matches": {
                    str(tol): len(pair_frames(*frames, tol)) for tol in TOLERANCES
                },
            }
    by_class = {}
    for kind in CLASSES:
        rows = [per_clip[key][kind] for key in paired]
        counts = {
            method: distribution([row[method] for row in rows]) for method in METHODS
        }
        totals = {method: counts[method]["events"] for method in METHODS}
        by_class[kind] = {}
        for tol in map(str, TOLERANCES):
            matches = sum(row["matches"][tol] for row in rows)
            total = sum(totals.values())
            by_class[kind][tol] = {
                "pairs": matches,
                "dice_agreement": 2 * matches / total if total else None,
                "fraction_of_events_matched": {
                    method: matches / n if n else None for method, n in totals.items()
                },
            }
        by_class[kind]["count_distribution"] = counts
        by_class[kind]["both_zero_clips"] = sum(
            not sum(row[m] for m in METHODS) for row in rows
        )
    return {
        "eligible_clips": len(keys),
        "paired_clips": len(paired),
        "unpaired_clips": sorted(set(keys) - set(paired)),
        "classes": by_class,
        "per_clip": per_clip,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--rule-root", type=Path, required=True)
    parser.add_argument("--labels-parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-only", action="store_true")
    args = parser.parse_args()

    manifest = read_json(args.benchmark / "manifest.json", PINS["manifest_sha256"])
    selection = read_json(args.selection, PINS["selection_sha256"])
    if (
        manifest["benchmark_id"] != BENCHMARK_ID
        or selection["manifest_sha256"] != PINS["manifest_sha256"]
    ):
        raise ValueError("Benchmark and selection disagree")
    clips = {clip["clip_id"]: clip for clip in manifest["clips"]}
    (fps,) = {clip["fps"] for clip in clips.values()}
    roles = {row["clip_id"]: row["role"] for row in selection["clips"]}
    if set(roles) != set(PINS["label_sha256"]) or Counter(roles.values()) != {
        "scoring": 6,
        "tuning": 2,
    }:
        raise ValueError("Unexpected evaluation split")

    progress_paths = {
        "refiner3": args.output / "refiner-progress.json",
        "rule": args.rule_root / "progress.json",
    }
    progress = {
        "refiner3": read_json(progress_paths["refiner3"]),
        "rule": read_json(progress_paths["rule"], PINS["rule_progress_sha256"]),
    }
    roots = {"refiner3": args.output, "rule": args.rule_root}
    predictions: dict[str, dict[str, Row]] = {method: {} for method in METHODS}
    hashes: dict[str, Any] = {
        "input_pins": PINS,
        "progress_sha256": {
            method: sha256(path) for method, path in progress_paths.items()
        },
        "prediction_sha256": {method: {} for method in METHODS},
    }
    coverage: dict[str, Row] = {key: {"role": roles.get(key)} for key in clips}
    for method, receipt in progress.items():
        version = (
            PINS["checkpoint_sha256"] if method == "refiner3" else PINS["rule_commit"]
        )
        if (
            receipt["method"] != method
            or receipt["method_version"] != version
            or receipt["manifest_sha256"] != PINS["manifest_sha256"]
            or set(receipt["clips"]) != set(clips)
            or receipt["total"] != len(clips)
            or Counter(item["status"] for item in receipt["clips"].values())
            != {
                status: receipt[status]
                for status in ("ok", "failed", "pending")
                if receipt[status]
            }
        ):
            raise ValueError(f"Invalid batch receipt: {method}")
        for key, row in receipt["clips"].items():
            coverage[key][method] = row
            if row["status"] == "pending":
                if not args.selected_only or key in roles:
                    raise ValueError(f"Prediction pending: {method} {key}")
                continue
            path = roots[method] / row["predictions"]
            predictions[method][key] = load_prediction(path, clips[key], row, method)
            hashes["prediction_sha256"][method][key] = sha256(path)

    by_split: dict[str, Row] = {
        role: {method: {kind: {} for kind in CLASSES} for method in METHODS}
        for role in ("scoring", "tuning")
    }
    details: dict[str, Row] = {}
    holdouts = {}
    for key, role in roles.items():
        clip = clips[key]
        label = load_labels(args.labels_parent, clip, role)
        masked = label_mask(label)
        holdouts[key] = {
            "role": role,
            "unlabelable_intervals": label["unlabelable"],
            "unlabelable_frames": len(masked),
            "label_status_counts": Counter(
                event["status"] for event in label["events"]
            ),
            "unavailable": {},
        }
        ball, valid = ball_context(args.benchmark, clip)
        details[key] = {"role": role, "point_spans_from_labels": point_spans(label)}
        for method in METHODS:
            clip_prediction = predictions[method][key]
            if clip_prediction["status"] == "failed":
                holdouts[key]["unavailable"][method] = {
                    "reason": clip_prediction["reason"],
                    "eligible_labels_skipped": Counter(
                        event["kind"]
                        for event in label["events"]
                        if event["status"] != "uncertain"
                        and event["frame"] not in masked
                    ),
                }
                continue
            events = clip_prediction["events"]
            details[key][method] = {}
            for kind in CLASSES:
                truth = [event for event in label["events"] if event["kind"] == kind]
                candidates = [event for event in events if event["kind"] == kind]
                for tolerance in TOLERANCES:
                    result = score_events(truth, candidates, masked, tolerance)
                    by_split[role][method][kind].setdefault(str(tolerance), []).append(
                        result
                    )
                    if tolerance == PRIMARY_TOLERANCE:
                        details[key][method][kind] = {
                            **result,
                            "contexts": error_tags(result, label, events, ball, valid),
                        }

    summary = {}
    for role, methods in by_split.items():
        summary[role] = {}
        for method, kinds in methods.items():
            summary[role][method] = {}
            for kind, results in kinds.items():
                summary[role][method][kind] = {
                    str(tol): report(results.get(str(tol), []), fps)
                    for tol in TOLERANCES
                }
                counts = {"fp": Counter(), "fn": Counter()}
                for value in details.values():
                    if value["role"] == role and method in value:
                        for side, tags in value[method][kind]["contexts"][
                            "counts"
                        ].items():
                            counts[side].update(tags)
                summary[role][method][kind]["mismatch_context_counts_at_5_frames"] = (
                    counts
                )

    unlabeled = [key for key in clips if key not in roles]
    metrics = {
        "benchmark": BENCHMARK_ID,
        "fps": fps,
        "primary_tolerance_frames": PRIMARY_TOLERANCE,
        "tolerances_frames": list(TOLERANCES),
        "scope": "selected_only" if args.selected_only else "all",
        "selected_clips": holdouts,
        "processing": {
            method: {k: value[k] for k in ("ok", "failed", "pending", "total")}
            for method, value in progress.items()
        },
        **summary,
    }
    save_json(args.output / "metrics.json", metrics)
    save_json(args.output / "matches-at-5-frames.json", details)
    save_json(args.output / "processing.json", coverage)
    save_json(
        args.output / "unlabeled-agreement.json", agreement(unlabeled, predictions)
    )
    save_json(args.output / "artifact-hashes.json", hashes)
    for role in ("scoring", "tuning"):
        for method in METHODS:
            for kind in CLASSES:
                row = metrics[role][method][kind][str(PRIMARY_TOLERANCE)]
                print(
                    role,
                    method,
                    kind,
                    *(
                        f"{k}={row[k]}"
                        for k in (
                            "tp",
                            "fp",
                            "fn",
                            "precision",
                            "recall",
                            "f1",
                            "mean_abs_error_ms",
                        )
                    ),
                )
    print(f"Saved {metrics['scope']} comparison: {args.output}")


if __name__ == "__main__":
    main()
