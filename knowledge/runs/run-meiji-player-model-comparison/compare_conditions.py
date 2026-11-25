"""Join pinned player accuracy and batch-one timing receipts into a comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.tennis_scene.evaluation.player_pose.bundle import pinned_file
from src.utils.inference_benchmark_inputs import PinnedBenchmarkInputs, TimingSchedule
from src.utils.inference_parity import PARITY_RULES
from src.utils.io import write_json_atomic

REPO_ROOT = Path(__file__).resolve().parents[3]
PT32_SCOPE = "対象外（ユーザー判断 2026-11-25）"
BACKENDS = ("pt32", "pt16", "trt16")
FP32_ATTENTION = "trt16-fp32attn"
SPEED_BATCH_SIZE = 1
HOST_THREADS = (4, 1)
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
PARITY_LABELS = {
    "iou_p05": "IoU P05",
    "score_abs_p95": "score error P95",
    "missing_ratio": "missing ratio",
    "extra_ratio": "extra ratio",
    "pose_xy_px_p95": "joint error P95 (px)",
    "pose_xy_px_max": "max joint error (px)",
    "pose_confidence_abs_p95": "joint score error P95",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_bytes())


def record_bytes(root: Path, record: dict[str, Any]) -> bytes:
    path = pinned_file(root, record)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError(f"Report SHA-256 differs from the pin: {record['path']}")
    return data


def record_json(root: Path, record: dict[str, Any]) -> Any:
    return json.loads(record_bytes(root, record))


def summarize_accuracy(
    document: dict[str, Any],
    row: dict[str, Any],
    cohort: dict[str, Any],
    *,
    verified_scorer_commit: str | None = None,
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
    reported_scorer = document.get("scorer_commit", verified_scorer_commit)
    if (
        reported_scorer != cohort["scorer_commit"]
        or (
            verified_scorer_commit is not None
            and verified_scorer_commit != reported_scorer
        )
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
    report: dict[str, Any],
    row: dict[str, Any],
    input_sha256: str,
    timing_sha256: str,
    schedule: TimingSchedule,
    *,
    verified_opencv_threads: int | None = None,
) -> dict[str, Any]:
    case, protocol = report["case"], report["protocol"]
    settings = case.get("settings", {})
    model_id, backend = row["model_id"], row["backend"]
    variant = backend == FP32_ATTENTION
    workload = (
        f"{model_id}/{backend}" if variant else WORKLOAD_NAMES.get(model_id, model_id)
    )
    report_backend = "trt16" if variant else backend
    # Successful reports prove the input order with IDs in every raw sample.
    order_digest_conflicts = (
        "timed_input_ids_sha256" in settings
        and settings["timed_input_ids_sha256"] != schedule.timed_ids_sha256
    )
    if (
        report["schema_version"] != 3
        or case["name"] != workload
        or case["backend"].lower() != report_backend
        or backend == "pt32"
        or (variant and settings.get("engine_condition") != backend)
        or case["batch_size"] != SPEED_BATCH_SIZE
        or case["item_unit"] != "image"
        or case["input_manifest_sha256"] != input_sha256
        or settings.get("timing_protocol_sha256") != timing_sha256
        or order_digest_conflicts
        or {**protocol, "batch_sizes": tuple(protocol["batch_sizes"])}
        != asdict(schedule.protocol)
    ):
        raise ValueError(
            f"Timing protocol or identity mismatch for {row['model_id']} {row['backend']}"
        )
    reason = report.get("parity_not_evaluated_reason")
    if report["status"] != "ok":
        raise ValueError(
            "Timing did not complete. Provide pinned terminal failure evidence"
        )
    if (
        report.get("parity_status") != "not_evaluated"
        or not isinstance(reason, str)
        or not reason.strip()
        or report.get("parity") is not None
    ):
        raise ValueError("Current timing requires explicit PT32 parity exclusion")
    opencv_threads = settings.get("opencv_threads", verified_opencv_threads)
    if (settings.get("host_threads"), opencv_threads) != HOST_THREADS or (
        verified_opencv_threads is not None
        and verified_opencv_threads != opencv_threads
    ):
        raise ValueError(
            f"Timing host thread settings differ from PyTorch 4 and OpenCV 1: {model_id}"
        )
    environment = report["environment"]
    if environment["gpu"] != "NVIDIA GeForce RTX 5060 Ti" or set(
        environment["fp32_precision"].values()
    ) != {"ieee"}:
        raise ValueError(f"Timing GPU or TF32 policy mismatch for {row['model_id']}")
    samples = report["samples"]
    iterations = schedule.protocol.iterations_per_block
    batches = schedule.batch_ids
    if len(samples) != iterations * schedule.protocol.blocks or any(
        tuple(sample["ids"]) != batches[i % len(batches)]
        or sample["items"] != SPEED_BATCH_SIZE
        or sample["block"] != 1 + i // iterations
        for i, sample in enumerate(samples)
    ):
        raise ValueError(
            f"Timing sample order differs from the speed manifest for {row['model_id']}"
        )
    wall, compute = (report["summary"][name] for name in ("stage_wall", "model_gpu"))
    return {
        "parity_status": "PT32未計測",
        "parity_reason": reason,
        "speed_status": "ok",
        "speed_batch_size": SPEED_BATCH_SIZE,
        "stage_p50_ms": wall["p50_ms_per_item"],
        "stage_p95_ms": wall["p95_ms_per_item"],
        "model_p50_ms": compute["p50_ms_per_item"],
        "model_p95_ms": compute["p95_ms_per_item"],
        "images_per_s": wall["items_per_second"],
        "nvml_peak_delta_mib": report["nvml_peak_minus_before_load_bytes"] / 1024**2,
        "torch_peak_allocated_mib": report["torch_peak_allocated_bytes"] / 1024**2,
        "host_threads": settings.get("host_threads"),
        "opencv_threads": opencv_threads,
    }


def verify_failure(
    artifacts: Path,
    row: dict[str, Any],
    failure: dict[str, Any],
    cohort_sha256: str,
    speed_sha256: str,
) -> None:
    log = failure["log"]
    lines = record_bytes(artifacts, log).decode("utf-8").splitlines()
    condition = condition_id(row)
    if (
        not lines
        or not re.fullmatch(r"exit_code=[1-9][0-9]*", lines[-1])
        or not any(line.startswith(f"{condition} infeasible (") for line in lines)
    ):
        raise ValueError(
            f"Terminal log does not record a failure for {condition}: {log['path']}"
        )
    report = record_json(artifacts, failure["report"])
    if (
        report["model_id"] != row["model_id"]
        or report["condition"] != row["backend"]
        or report["status"] != "infeasible"
        or report["accuracy_manifest_sha256"] != cohort_sha256
        or report["speed_manifest_sha256"] != speed_sha256
    ):
        raise ValueError(
            f"Failure report disagrees with the fixed condition: {condition}"
        )


def measured_points(rows, workload):
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


def saved_parity_fields(
    metrics: dict[str, Any],
    *,
    images: int,
    scope: str,
    min_score: float | None,
) -> dict[str, Any]:
    """Apply the shared thresholds to stored metrics with their original input scope."""
    failures = []
    values = {}
    for key, label in PARITY_LABELS.items():
        if key.startswith("pose_") and key not in metrics:
            continue
        value = metrics[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid saved PT32 comparison metric: {key}={value}")
        rule = PARITY_RULES[key]
        assert rule.limit is not None
        if rule.comparison == ">=" and value < rule.limit:
            failures.append(f"{label} {value:.4f} < {rule.limit:g}")
        if rule.comparison == "<=" and value > rule.limit:
            failures.append(f"{label} {value:.4f} > {rule.limit:g}")
        values[f"common_parity_{key}"] = value
    return {
        "common_parity_status": "failed" if failures else "passed",
        "common_parity_scope": scope,
        "common_parity_images": images,
        "common_parity_min_score": min_score,
        "common_parity_failures": ", ".join(failures),
        **values,
    }


def asset_hashes(report: dict[str, Any], name: str) -> dict[str, str]:
    return {key: item["sha256"] for key, item in report[name].items()}


def rfdetr_saved_parity(
    receipt: dict[str, Any],
    case: dict[str, Any],
    reference: dict[str, Any],
    timing: dict[str, Any],
    cohort: dict[str, Any],
) -> dict[str, Any]:
    backend = case["backend"]
    condition = receipt["conditions"][backend]
    baseline = receipt["conditions"]["pt32"]
    proof = condition["parity_vs_pt32"]
    metrics = proof["metrics"]
    engine_hashes = (
        {
            "plan": next(
                item["sha256"]
                for item in receipt["conversion"]["tensorrt_bundle"]
                if Path(item["path"]).suffix == ".plan"
            )
        }
        if backend == "trt16"
        else {}
    )
    if (
        receipt["model_id"] != case["model_id"]
        or receipt["benchmark_id"] != cohort["id"]
        or receipt["manifest_sha256"] != cohort["sha256"]
        or metrics["images"] != cohort["images"]
        or condition["artifacts"]["predictions"]["sha256"]
        != case["predictions"]["sha256"]
        or baseline["artifacts"]["predictions"]["sha256"]
        != reference["predictions"]["sha256"]
        or asset_hashes(timing, "weights") != {"model": receipt["weights"]["sha256"]}
        or asset_hashes(timing, "engines") != engine_hashes
    ):
        raise ValueError(
            f"Saved RF-DETR PT32 comparison disagrees with accuracy or timing: {backend}"
        )
    return {
        "historical_parity_status": "historical_passed"
        if proof["passed"]
        else "historical_failed",
        **saved_parity_fields(
            {
                "iou_p05": metrics["matched_iou_p05"],
                "score_abs_p95": metrics["abs_score_difference"]["p95"],
                "missing_ratio": 1 - metrics["pt32_retained_fraction"],
                "extra_ratio": 1 - metrics["candidate_matched_fraction"],
            },
            images=metrics["images"],
            scope="accuracy cohort",
            min_score=condition["settings"]["min_person_score_inclusive"],
        ),
    }


def dino_saved_parity(
    history: dict[str, Any], timing: dict[str, Any], image_count: int
) -> dict[str, Any]:
    old, new = history["case"], timing["case"]
    settings = (
        "dino_confidence",
        "dino_hw",
        "pose_crop_batch",
        "vitpose_hw",
        "flip_test",
    )
    proof = history["parity"]
    metrics = proof["metrics"]
    if (
        any(
            old[key] != new[key]
            for key in ("name", "backend", "batch_size", "input_manifest_sha256")
        )
        or any(old["settings"].get(key) != new["settings"].get(key) for key in settings)
        or any(
            asset_hashes(history, key) != asset_hashes(timing, key)
            for key in ("weights", "engines")
        )
        or metrics["checked_ids"] != image_count
    ):
        raise ValueError(
            f"Saved DINO PT32 comparison disagrees with the current condition: {new['name']}/{new['backend']}"
        )
    return {
        "historical_parity_status": "historical_passed"
        if proof["passed"]
        else "historical_failed",
        **saved_parity_fields(
            {
                **metrics,
                "missing_ratio": metrics["missing"]
                / max(metrics["reference_boxes"], 1),
                "extra_ratio": metrics["extra"] / max(metrics["candidate_boxes"], 1),
            },
            images=metrics["checked_ids"],
            scope="initial timing inputs",
            min_score=old["settings"]["dino_confidence"],
        ),
    }


def edgecrafter_saved_parity(
    receipt: dict[str, Any],
    cases: dict[tuple[str, str], dict[str, Any]],
    historical: list[dict[str, Any]],
    cohort: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    if (
        receipt["benchmark_id"] != cohort["id"]
        or receipt["manifest_sha256"] != cohort["sha256"]
    ):
        raise ValueError(
            "Saved EdgeCrafter PT32 comparison uses a different accuracy cohort"
        )
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    archived = {(case["model_id"], case["backend"]): case for case in historical}
    for proof in receipt["conditions"]:
        model, backend = key = proof["model_id"], proof["backend"]
        metrics = proof["metrics"]
        if (
            metrics["checked_ids"] != cohort["images"]
            or proof["pt32_predictions_sha256"]
            != cases[model, "pt32"]["predictions"]["sha256"]
        ):
            raise ValueError(
                f"Saved EdgeCrafter PT32 comparison has a different reference: {key}"
            )
        if proof["candidate_predictions_sha256"] != cases[key]["predictions"]["sha256"]:
            if proof["candidate_predictions_sha256"] != archived.get(key, {}).get(
                "predictions", {}
            ).get("sha256"):
                raise ValueError(
                    f"Saved EdgeCrafter PT32 comparison has unknown predictions: {key}"
                )
            continue
        row = saved_parity_fields(
            metrics, images=cohort["images"], scope="accuracy cohort", min_score=None
        )
        if (row["common_parity_status"] == "passed") != (proof["status"] == "ok"):
            raise ValueError(
                f"Saved EdgeCrafter PT32 comparison differs from the shared thresholds: {key}"
            )
        rows[key] = row
    return rows


def frontiers(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    result = {}
    for workload, metric in QUALITY_METRIC.items():
        points = [
            row
            for row in measured_points(rows, workload)
            if not row.get("pareto_exclusion_reason")
        ]
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
    schedule_pin = inputs["timing_protocol"]
    pinned = PinnedBenchmarkInputs(
        pinned_file(artifacts, inputs).parent,
        pinned_file(REPO_ROOT, schedule_pin),
        manifest_sha256=inputs["sha256"],
        timing_sha256=schedule_pin["sha256"],
    )
    schedule = pinned.schedule("frames", SPEED_BATCH_SIZE)
    if len(schedule.batch_ids) != 96 or pinned.manifest["id"] != inputs["id"]:
        raise ValueError("Unexpected speed input cohort")
    cases = {
        (case["model_id"], case["backend"]): case for case in evidence["conditions"]
    }
    planned = [
        (model, backend)
        for model in MODELS
        for backend in (*BACKENDS, FP32_ATTENTION)
        if backend != FP32_ATTENTION
        or model.startswith(("ecdet-", "ecpose-", "rtdetrv4-"))
    ]
    if len(cases) != len(evidence["conditions"]) or not cases.keys() <= set(planned):
        raise ValueError("Duplicate or unplanned comparison condition")
    for receipt in evidence.get("candidate_receipts", {}).values():
        for pin in receipt.values():
            if isinstance(pin, dict):
                pinned_file(REPO_ROOT, pin)
    for job in (
        *evidence.get("baseline_accuracy_jobs", {}).values(),
        evidence.get("joint_speed_run", {}),
    ):
        if "queue_log" in job:
            pinned_file(artifacts, job["queue_log"])
    if log_checksums := evidence.get("retained_queue_log_checksums"):
        pinned_file(artifacts, log_checksums)

    receipts = evidence.get("candidate_receipts", {})
    rf_pin = receipts.get("rf_detr", {}).get("accuracy")
    rf_receipt = record_json(REPO_ROOT, rf_pin) if rf_pin else None
    edge_pin = receipts.get("edgecrafter", {}).get("shared_parity")
    edge_parity = (
        edgecrafter_saved_parity(
            record_json(REPO_ROOT, edge_pin),
            cases,
            evidence.get("historical_conditions", []),
            cohort,
        )
        if edge_pin
        else {}
    )

    rows = []
    for model, backend in planned:
        row = {
            "model_id": model,
            "backend": backend,
            "workload": "detection_and_pose" if model in POSE_MODELS else "detection",
            "accuracy_status": "pending",
            "speed_status": "pending",
            "accuracy_batch_size": None,
            "speed_batch_size": None,
            "host_threads": None,
            "opencv_threads": None,
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
            rows.append(row)
            continue
        if failure := case.get("failure"):
            if "accuracy" in case or "timing" in case:
                raise ValueError(
                    f"Failure conflicts with a measurement for {condition_id(row)}"
                )
            verify_failure(artifacts, row, failure, cohort["sha256"], inputs["sha256"])
            row.update(
                accuracy_status="infeasible",
                speed_status="infeasible",
                accuracy_reason=failure["reason"],
                speed_reason=failure["reason"],
            )
        if reason := case.get("not_applicable_reason"):
            if "accuracy" in case or "timing" in case or "failure" in case:
                raise ValueError(
                    f"Applicability conflicts with a measurement for {condition_id(row)}"
                )
            row.update(
                accuracy_status="not_applicable",
                speed_status="not_applicable",
                accuracy_reason=reason,
                speed_reason=reason,
            )
        if unmatched := case.get("unmatched_timing"):
            if "timing" in case:
                raise ValueError(
                    f"Unmatched and matched timing selected for {condition_id(row)}"
                )
            pinned_file(artifacts, unmatched)
            row.update(speed_status="unmatched_host", speed_reason=case["speed_reason"])
        if backend == "pt32":
            row.update(speed_status="excluded", speed_reason=PT32_SCOPE)
            if "accuracy" not in case:
                row.update(accuracy_status="excluded", accuracy_reason=PT32_SCOPE)
        if "historical_timing" in case and "timing" not in case:
            row["speed_reason"] = case.get(
                "speed_reason",
                "Historical timing predates synchronized stage wall measurement",
            )
            if backend == "pt32":
                row["speed_status"] = "historical"
        if reason := case.get("parity_not_evaluated_reason"):
            row.update(parity_status="PT32未計測", parity_reason=reason)
        if "accuracy" in case:
            accuracy_pin = case["accuracy"]
            accuracy = record_json(artifacts, accuracy_pin)
            if accuracy["predictions_sha256"] != case["predictions"]["sha256"]:
                raise ValueError(
                    f"Accuracy has a different prediction digest for {model} {backend}"
                )
            batch_size = accuracy_pin["batch_size"]
            reported_batches = (
                accuracy.get("image_batch_size"),
                accuracy.get("batch_size"),
                accuracy.get("runtime", {}).get("batch_size"),
            )
            if (
                type(batch_size) is not int
                or batch_size < 1
                or any(
                    reported is not None and reported != batch_size
                    for reported in reported_batches
                )
            ):
                raise ValueError(
                    f"Accuracy batch size differs from the pin for {condition_id(row)}"
                )
            row.update(
                summarize_accuracy(
                    accuracy,
                    row,
                    cohort,
                    verified_scorer_commit=accuracy_pin.get("scorer_commit"),
                )
            )
            row.update(accuracy_status="measured", accuracy_batch_size=batch_size)
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
                row["historical_parity_status"] = row["parity_status"] = (
                    "historical_passed" if proof["passed"] else "historical_failed"
                )
        if "timing" in case:
            timing = record_json(artifacts, case["timing"])
            row.update(
                summarize_timing(
                    timing,
                    row,
                    inputs["sha256"],
                    schedule_pin["sha256"],
                    schedule,
                    verified_opencv_threads=case["timing"].get("opencv_threads"),
                )
            )
            if model in WORKLOAD_NAMES and "historical_timing" in case:
                history = record_json(artifacts, case["historical_timing"])
                row.update(dino_saved_parity(history, timing, len(schedule.batch_ids)))
            if model == "rfdetr-l" and rf_receipt is not None:
                row.update(
                    rfdetr_saved_parity(
                        rf_receipt, case, cases[model, "pt32"], timing, cohort
                    )
                )
        row.update(edge_parity.get((model, backend), {}))
        if reason := case.get("pareto_exclusion_reason"):
            row["pareto_exclusion_reason"] = reason
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


def saved_pt32_note(row: dict[str, Any]) -> str:
    score = row["common_parity_min_score"]
    cutoff = "all saved scores" if score is None else f"score ≥ {score:.2f}"
    observed = (
        row["common_parity_failures"] or f"IoU P05 {row['common_parity_iou_p05']:.4f}"
    )
    return (
        f"{condition_id(row)}  ·  {row['common_parity_status']}  ·  "
        f"{row['common_parity_images']:,} images, {row['common_parity_scope']}, {cutoff}  ·  {observed}"
    )


def plot_pareto(directory: Path, result: dict[str, Any]) -> None:
    for workload, metric in QUALITY_METRIC.items():
        points = measured_points(result["rows"], workload)
        if not points:
            continue
        from matplotlib.figure import Figure

        front = set(result["pareto"][workload])
        ordered_front = sorted(
            (point for point in points if condition_id(point) in front),
            key=lambda point: point["stage_p50_ms"],
        )
        detection = workload == "detection"
        without_speed = [
            row
            for row in result["rows"]
            if row["workload"] == workload
            and row["speed_status"]
            in {"unmatched_host", "infeasible", "not_applicable"}
        ]
        saved_notes = [
            saved_pt32_note(row)
            for row in result["rows"]
            if row["workload"] == workload and row.get("common_parity_status")
        ]
        chart_rows = 2 if detection else 1
        chart_height = 6.5 if detection else 4.8
        note_height = 0.20 * (len(saved_notes) + 1) if saved_notes else 0
        figure = Figure(
            figsize=(13.5 if without_speed else 9, chart_height + note_height),
            layout="constrained",
        )
        grid = figure.add_gridspec(
            chart_rows + bool(saved_notes),
            2 if without_speed else 1,
            width_ratios=[2.0, 1.0] if without_speed else [1.0],
            height_ratios=[chart_height / chart_rows] * chart_rows
            + ([note_height] if saved_notes else []),
        )
        axes = [figure.add_subplot(grid[0, 0])]
        if detection:
            axes.append(figure.add_subplot(grid[1, 0], sharex=axes[0]))
            axes[0].tick_params(labelbottom=False)
        front_x = [point["stage_p50_ms"] for point in ordered_front]
        front_y = [100 * point[metric] for point in ordered_front]
        ylabel = (
            "Recall at IoU 0.50 (%)" if detection else "Mean coverage-inclusive OKS (%)"
        )
        for axis in axes:
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
            axis.plot(front_x, front_y, color="#1763a6", alpha=0.55, linewidth=1)
            for point in points:
                case = condition_id(point)
                excluded = point.get("pareto_exclusion_reason")
                color = (
                    "#a44827" if excluded else "#1763a6" if case in front else "#6e7882"
                )
                axis.scatter(
                    point["stage_p50_ms"],
                    100 * point[metric],
                    color=color,
                    marker="x" if excluded else "o",
                    s=34,
                )
            axis.margins(x=0.10)

        axes[-1].set_ylim(-4, 104)
        axes[-1].set_xlabel("Batch 1 stage wall time (ms per image, P50)")
        labelled_axis = axes[0]
        if detection and front_y:
            labelled_axis.set_ylim(min(front_y) - 0.6, max(front_y) + 0.6)
            labelled_axis.set_title(
                "Detail near the Pareto frontier", loc="left", fontsize=10
            )
        labels = ordered_front + [
            point
            for point in points
            if point["model_id"] == "dino-swin-l" and condition_id(point) not in front
        ]
        for index, point in enumerate(labels):
            x, y = point["stage_p50_ms"], 100 * point[metric]
            bottom = index % 2 == 0
            align_right = x > sum(axes[-1].get_xlim()) / 2
            labelled_axis.annotate(
                condition_id(point),
                (x, y),
                xytext=(-6 if align_right else 6, -14 if bottom else 6),
                ha="right" if align_right else "left",
                textcoords="offset points",
                fontsize=8,
            )
        if without_speed:
            diagnostic_axis = figure.add_subplot(grid[:chart_rows, 1])
            diagnostic_axis.set_title(
                "Plans without matched speed", loc="left", fontsize=10
            )
            labels = [condition_id(row) for row in without_speed]
            diagnostic_axis.set_yticks(range(len(labels)), labels, fontsize=8)
            diagnostic_axis.set_ylim(len(labels) - 0.5, -0.5)
            diagnostic_axis.set_xlim(-4, 104)
            diagnostic_axis.set_xlabel(ylabel)
            diagnostic_axis.grid(axis="x", alpha=0.25)
            for index, row in enumerate(without_speed):
                quality = row.get(metric)
                if quality is None:
                    status = (
                        "Not applicable"
                        if row["speed_status"] == "not_applicable"
                        else "Infeasible · raw outputs NaN"
                    )
                    diagnostic_axis.text(
                        0.05,
                        index,
                        status,
                        va="center",
                        fontsize=8,
                        transform=diagnostic_axis.get_yaxis_transform(),
                    )
                else:
                    diagnostic_axis.scatter(
                        100 * quality, index, color="#a44827", marker="x", s=34
                    )

        if saved_notes:
            notes = figure.add_subplot(grid[chart_rows, :])
            notes.set_axis_off()
            notes.text(
                0,
                1,
                "\n".join(
                    [
                        "Saved PT32 output checks · frontier uses pseudo-GT accuracy and stage wall time",
                        *saved_notes,
                    ]
                ),
                va="top",
                fontsize=8.5,
                linespacing=1.4,
            )

        accuracy_batches = sorted(
            {
                row["accuracy_batch_size"]
                for row in points + without_speed
                if row.get("accuracy_batch_size") is not None
            }
        )
        batches = ", ".join(map(str, accuracy_batches))
        accuracy_label = f"Accuracy at image batch {batches}"
        if 2 not in accuracy_batches:
            accuracy_label += " · batch 2 accuracy unmeasured"
        figure.suptitle(
            "Speed and VRAM at image batch 1 · PyTorch 4 / OpenCV 1 CPU threads\n"
            f"{accuracy_label} · × conversion fault, cause unknown",
            fontsize=11,
        )
        figure.savefig(directory / f"pareto-{workload}.png", dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    result = comparison(args.artifacts, read_json(args.evidence))
    write_json_atomic(args.output / "comparison.json", result)
    write_csv(args.output / "conditions.csv", result["rows"])
    if not args.tables_only:
        plot_pareto(args.output, result)


if __name__ == "__main__":
    main()
