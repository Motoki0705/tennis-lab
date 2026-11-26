"""Verify the paired predictions and promote only the comparison summaries."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from evaluate import CONDITIONS, MODELS, sha, summarize, write_json

from src.tasks.ball_detection.evaluation.temporal_heatmap_metrics import calibrate

POSITION_METRICS = ("mean_error_px", "median_error_px", "p95_error_px", "within_distance_rate")
PRESENCE_METRICS = ("negative_fpr", "observed_fnr", "false_positive", "false_negative")
SUPPORT = ("frames", "observed", "reviewed_empty", "out_of_frame")


def check_frames(records, anchors):
    if len(records) != len(anchors):
        raise ValueError("Prediction support does not match the anchor plan")
    for record, anchor in zip(records, anchors, strict=True):
        if any(record[key] != value for key, value in anchor.items()):
            raise ValueError(f"Prediction is attached to a different anchor: {anchor['clip_id']}/{anchor['frame_index']}")
        x, y = record["pred_xy"]
        score = record["score"]
        if not all(math.isfinite(v) for v in (x, y, score)) or not 0 <= score <= 1:
            raise ValueError("Prediction must have finite coordinates and a valid score")
        dx, dy = (None, None) if anchor["teacher_xy"] is None else (x - anchor["teacher_xy"][0], y - anchor["teacher_xy"][1])
        if dx is None:
            if record["error_px"] is not None:
                raise ValueError("Presence-only reference has a location error")
        elif not math.isclose(record["error_px"], math.hypot(dx, dy), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Stored location error differs from the source-pixel coordinates")


def check_metrics(records, anchors, metrics, calibration):
    check_frames(records, anchors)
    measured = summarize(records, calibration)
    if metrics != measured:
        raise ValueError("Stored metrics do not reproduce from the recorded predictions")
    scores = [record["score"] for record in records]
    return measured | {"score_min": min(scores), "score_max": max(scores)}


def paired_hits(normal, changed):
    counts = Counter(
        (a["error_px"] <= 20, b["error_px"] <= 20)
        for a, b in zip(normal, changed, strict=True) if a["kind"] == "observed"
    )
    return dict(
        both=counts[True, True], ordered_only=counts[True, False],
        changed_only=counts[False, True], neither=counts[False, False],
    )


def write_table(path: Path, models: dict) -> None:
    rows = []
    for name, record in models.items():
        for condition, metrics in record["metrics"].items():
            for source, subset in {"all": metrics, **metrics["by_source"]}.items():
                rows.append(dict(
                    model=name, condition=condition, source=source,
                    **{field: subset["support"][field] for field in SUPPORT},
                    **{field: subset[field] for field in (*POSITION_METRICS, *PRESENCE_METRICS)},
                ))
    with path.open("w", newline="", encoding="utf-8") as stream:
        table = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        table.writeheader()
        table.writerows(rows)


def collect(anchors_path: Path, shape_path: Path, temporal_path: Path):
    shape, temporal = [json.loads(path.read_bytes()) for path in (shape_path, temporal_path)]
    plan_sha = sha(anchors_path)
    if plan_sha != shape["plan_sha256"] or plan_sha != temporal["plan_sha256"]:
        raise ValueError("Conditions used different plans")
    plan = json.loads(anchors_path.read_bytes())
    anchors = plan["anchors"]
    if len({(r["clip_id"], r["frame_index"]) for r in anchors}) != 264:
        raise ValueError("Anchor plan contains a duplicate or omission")
    evaluation = [r for r in anchors if r["split"] == "evaluation"]
    calibration = [r for r in anchors if r["split"] == "calibration"]
    if (
        Counter(r["kind"] for r in evaluation) != {"observed": 88, "reviewed_empty": 32, "out_of_frame": 24}
        or len(calibration) != 120 or len(anchors) != 264
        or {r["source_group"] for r in evaluation} & {r["source_group"] for r in calibration}
    ):
        raise ValueError("Unexpected evaluation support or overlap with calibration")
    shape_cal = shape["predictions"]["calibration"]
    check_frames(shape_cal, calibration)
    baseline = check_metrics(shape["predictions"]["evaluation"], evaluation, shape["metrics"], calibrate(shape_cal, split="val"))

    models = {}
    if set(temporal["models"]) != set(MODELS):
        raise ValueError("Incomplete model results")
    for name, item in temporal["models"].items():
        expected = MODELS[name] | {"size": list(MODELS[name]["size"])}
        if expected != item["checkpoint"] or set(item["predictions"]) != set(CONDITIONS):
            raise ValueError(f"Unexpected model configuration or incomplete conditions: {name}")
        threshold = dict(mode="threshold", threshold=MODELS[name]["threshold"])
        normal = item["predictions"]["ordered"]
        metrics = {}
        for condition in CONDITIONS:
            records = item["predictions"][condition]
            metrics[condition] = check_metrics(records, evaluation, item["metrics"][condition], threshold)
        models[name] = dict(
            checkpoint=item["checkpoint"], seconds=item["seconds"],
            peak_cuda_allocated_bytes=item["peak_cuda_allocated_bytes"], metrics=metrics,
            paired_hits_20_px={
                condition: paired_hits(normal, item["predictions"][condition])
                for condition in CONDITIONS if condition != "ordered"
            },
        )
    models["shape_baseline"] = dict(metrics={"color_roundness_motion": baseline})
    return dict(
        plan_sha256=plan_sha, manifest_sha256=plan["manifest_sha256"],
        raw_shape_sha256=sha(shape_path), raw_temporal_sha256=sha(temporal_path),
        calibration_anchors=len(calibration), evaluation_anchors=len(evaluation),
        torch=temporal["torch"], opencv=temporal["opencv"], gpu=temporal["gpu"], models=models,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("anchors", "shape", "temporal", "output"):
        parser.add_argument(f"--{arg}", required=True, type=Path)
    args = parser.parse_args()
    report = collect(args.anchors, args.shape, args.temporal)
    write_json(args.output / "metrics.json", report)
    write_table(args.output / "comparison.csv", report["models"])
    print("Verified 6 GPU conditions and 1 CPU condition on the same 144 anchors, plus 120 calibration anchors")


if __name__ == "__main__":
    main()
