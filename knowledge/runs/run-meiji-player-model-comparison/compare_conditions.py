"""Join pinned player accuracy and batch-one timing receipts into a comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.tennis_scene.evaluation.player_pose.bundle import pinned_file
from src.utils.io import write_json_atomic

BACKENDS = ("pt32", "pt16", "trt16")
MODELS = (
    "dino-swin-l",
    "dino-swin-l-vitpose-h",
    "rfdetr-l",
    "rfdetr-2xl",
    "ecdet-l-coco",
    "ecdet-x-coco",
    "ecdet-l-o365",
    "ecdet-x-o365",
    "rtdetrv4-l",
    "rtdetrv4-x",
    "ecpose-l-o365",
    "ecpose-x-o365",
)
WORKLOAD_NAMES = {"dino-swin-l": "dino", "dino-swin-l-vitpose-h": "dino-vitpose"}
POSE_MODELS = {"dino-swin-l-vitpose-h", "ecpose-l-o365", "ecpose-x-o365"}
QUALITY_METRIC = {"detection": "recall_50", "detection_and_pose": "coverage_oks"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_bytes())


def record_json(root: Path, record: dict[str, Any]) -> Any:
    return read_json(pinned_file(root, record))


def summarize_accuracy(
    document: dict[str, Any], row: dict[str, Any], cohort: dict[str, Any]
) -> dict[str, Any]:
    header = document.get("condition", document)
    expected = {
        "model_id": row["model_id"],
        "backend": row["backend"],
        "has_pose": row["workload"] == "detection_and_pose",
        "benchmark_id": cohort["id"],
        "manifest_sha256": cohort["sha256"],
    }
    if any(header[key] != value for key, value in expected.items()):
        raise ValueError(
            f"Accuracy identity mismatch for {row['model_id']} {row['backend']}"
        )
    score = document["score"]
    detection = score["overall"]["detection"]
    if (
        document["scorer_commit"] != cohort["scorer_commit"]
        or score["images"] != cohort["images"]
        or detection["references"] != cohort["eligible_boxes"]
    ):
        raise ValueError(
            f"Accuracy disagrees with the frozen scorer or cohort for {row['model_id']}"
        )
    pose = score["overall"]["pose"]["with_detection_coverage"]
    return {
        "recall_50": detection["recall_by_iou"]["0.50"],
        "recall_75": detection["recall_by_iou"]["0.75"],
        "recall_50_score_ge_0_5": detection["recall_by_min_score"]["0.5"]["0.50"],
        "small_recall_50": score["by_size"]["small"]["detection"]["recall_by_iou"][
            "0.50"
        ],
        "coverage_oks": pose["oks"]["mean"] if expected["has_pose"] else None,
        "coverage_pck_0_10": pose["pck"]["0.1"] if expected["has_pose"] else None,
        "joint_coverage": pose["valid_joint_fraction"]
        if expected["has_pose"]
        else None,
    }


def summarize_timing(
    report: dict[str, Any], row: dict[str, Any], input_sha256: str, frame_ids: list[str]
) -> dict[str, Any]:
    case, protocol = report["case"], report["protocol"]
    if (
        case["name"] != WORKLOAD_NAMES.get(row["model_id"], row["model_id"])
        or case["backend"].lower() != row["backend"]
        or case["batch_size"] != 1
        or case["item_unit"] != "image"
        or case["input_manifest_sha256"] != input_sha256
        or (protocol["warmup"], protocol["iterations_per_block"], protocol["blocks"])
        != (30, 100, 3)
    ):
        raise ValueError(
            f"Timing protocol or identity mismatch for {row['model_id']} {row['backend']}"
        )
    if report["status"] != "ok":
        return {
            "speed_status": report["status"],
            "speed_reason": report["failure"]["message"],
        }
    environment = report["environment"]
    if environment["gpu"] != "NVIDIA GeForce RTX 5060 Ti" or set(
        environment["fp32_precision"].values()
    ) != {"ieee"}:
        raise ValueError(f"Timing GPU or TF32 policy mismatch for {row['model_id']}")
    samples = report["samples"]
    if len(samples) != 300 or any(
        sample["ids"] != [frame_ids[i % len(frame_ids)]]
        or sample["block"] != 1 + i // 100
        for i, sample in enumerate(samples)
    ):
        raise ValueError(
            f"Timing sample order differs from the speed manifest for {row['model_id']}"
        )
    wall, compute = (report["summary"][name] for name in ("stage_wall", "model_gpu"))
    return {
        "speed_status": "ok",
        "stage_p50_ms": wall["p50_ms_per_item"],
        "stage_p95_ms": wall["p95_ms_per_item"],
        "model_p50_ms": compute["p50_ms_per_item"],
        "model_p95_ms": compute["p95_ms_per_item"],
        "images_per_s": wall["items_per_second"],
        "nvml_peak_delta_mib": report["nvml_peak_minus_before_load_bytes"] / 1024**2,
        "torch_peak_allocated_mib": report["torch_peak_allocated_bytes"] / 1024**2,
    }


def qualified_points(rows, workload):
    metric = QUALITY_METRIC[workload]
    return [
        row
        for row in rows
        if row["workload"] == workload
        and row["accuracy_status"] == "measured"
        and row["speed_status"] == "ok"
        and row.get(metric) is not None
    ]


def condition_id(row):
    return f"{row['model_id']}/{row['backend']}"


def frontiers(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    result = {}
    for workload, metric in QUALITY_METRIC.items():
        points = qualified_points(rows, workload)
        result[workload] = [
            condition_id(point)
            for point in points
            if not any(
                other["stage_p50_ms"] <= point["stage_p50_ms"]
                and other[metric] >= point[metric]
                and (
                    other["stage_p50_ms"] < point["stage_p50_ms"]
                    or other[metric] > point[metric]
                )
                for other in points
            )
        ]
    return result


def comparison(artifacts: Path, evidence: dict[str, Any]) -> dict[str, Any]:
    cohort, inputs = evidence["evaluation"], evidence["speed_input"]
    manifest = pinned_file(artifacts, inputs)
    manifest_doc = read_json(manifest)
    index = manifest_doc["indexes"]["frames"]
    frames = record_json(manifest.parent, index)
    frame_ids = [frame["id"] for frame in frames]
    if (
        len(frame_ids) != 96
        or len(set(frame_ids)) != 96
        or manifest_doc["id"] != inputs["id"]
    ):
        raise ValueError("Unexpected speed input cohort")
    cases = {
        (case["model_id"], case["backend"]): case for case in evidence["conditions"]
    }
    planned = [(model, backend) for model in MODELS for backend in BACKENDS]
    if len(cases) != len(evidence["conditions"]) or not cases.keys() <= set(planned):
        raise ValueError("Duplicate or unplanned comparison condition")
    rows = []
    for model, backend in planned:
        row = {
            "model_id": model,
            "backend": backend,
            "workload": "detection_and_pose" if model in POSE_MODELS else "detection",
            "accuracy_status": "pending",
            "speed_status": "pending",
        }
        case = cases.get((model, backend), {})
        if model == "rfdetr-2xl":
            if case:
                raise ValueError("RF-DETR-2XL is excluded by the license decision")
            row.update(
                accuracy_status="excluded",
                speed_status="excluded",
                accuracy_reason="RF-DETR-2XL excluded by the 2026-11-24 license decision",
            )
        if "accuracy" in case:
            accuracy = record_json(artifacts, case["accuracy"])
            if accuracy["predictions_sha256"] != case["predictions"]["sha256"]:
                raise ValueError(
                    f"Accuracy has a different prediction digest for {model} {backend}"
                )
            row.update(summarize_accuracy(accuracy, row, cohort))
            row["accuracy_status"] = "measured"
            if "parity" in case:
                proof = record_json(artifacts, case["parity"])
                if (
                    proof["candidate_predictions_sha256"]
                    != accuracy["predictions_sha256"]
                    or proof["model_id"] != model
                    or proof["candidate_backend"] != backend
                    or proof["manifest_sha256"] != cohort["sha256"]
                ):
                    raise ValueError(
                        f"Parity evidence differs from accuracy for {model} {backend}"
                    )
                if proof["passed"] is False:
                    row.update(
                        accuracy_status="parity_failed", accuracy_reason=proof["reason"]
                    )
        if "timing" in case:
            row.update(
                summarize_timing(
                    record_json(artifacts, case["timing"]),
                    row,
                    inputs["sha256"],
                    frame_ids,
                )
            )
        rows.append(row)
    return {
        "evaluation": cohort,
        "speed_input": inputs,
        "rows": rows,
        "pareto": frontiers(rows),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = tuple(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_pareto(directory: Path, result: dict[str, Any]) -> None:
    for workload, metric in QUALITY_METRIC.items():
        points = qualified_points(result["rows"], workload)
        if not points:
            continue
        from matplotlib.figure import Figure

        front = set(result["pareto"][workload])
        figure = Figure(figsize=(9, 5), layout="constrained")
        axis = figure.subplots()
        for point in points:
            case = condition_id(point)
            x, y = point["stage_p50_ms"], 100 * point[metric]
            color = "#1763a6" if case in front else "#6e7882"
            axis.scatter(x, y, color=color, s=34)
            if case in front:
                axis.annotate(
                    case, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8
                )
        axis.margins(x=0.15, y=0.15)
        axis.set_xlabel("Batch 1 stage wall time (ms per image, P50)")
        axis.set_ylabel(
            "Recall at IoU 0.50 (%)"
            if workload == "detection"
            else "Mean coverage-inclusive OKS (%)"
        )
        axis.grid(alpha=0.25)
        figure.savefig(directory / f"pareto-{workload}.png", dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = comparison(args.artifacts, read_json(args.evidence))
    write_json_atomic(args.output / "comparison.json", result)
    write_csv(args.output / "conditions.csv", result["rows"])
    plot_pareto(args.output, result)


if __name__ == "__main__":
    main()
