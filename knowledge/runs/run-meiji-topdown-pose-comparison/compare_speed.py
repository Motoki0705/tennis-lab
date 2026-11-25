"""Join retained pose accuracy and synchronized speed, then plot the trade-offs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from speed_inputs import (
    INPUT_ID,
    INPUT_MANIFEST_SHA256,
    TIMING_PROTOCOL,
    TIMING_PROTOCOL_SHA256,
)

from src.utils.checksum import dual_sha256
from src.utils.inference_benchmark_inputs import PinnedBenchmarkInputs, TimingSchedule

RUN = Path(__file__).parent
BASELINE = RUN.parent / "run-person-model-speed-low-precision" / "cases.json"
REFERENCE_MODEL = "vitpose-h"
MODELS = (REFERENCE_MODEL, "rtmpose-m", "rtmpose-x", "pmpose-l", "pmpose-h")
BACKENDS = ("PT16", "TRT16")
BATCH_SIZES = (1, 4, 8)
PARITY_LABEL = "PT32未計測"
HARNESS_MERGE = "59faa32a28d2f54e6492abdba2429846fa0397e3"
ACCURACY_SHA256 = "3dc496513380a39d4511ff7f8711f4192eababa5109f6a0506f88a60baa70580"
DEVICE_KEYS = ("gpu", "gpu_uuid", "driver", "cuda", "torch")

CaseKey = tuple[str, str, int]


@dataclass(frozen=True)
class Point:
    mean_oks: float
    wall_ms_per_person: float
    nvml_peak_bytes: int

    def costs(self) -> tuple[float, float, int]:
        return (-self.mean_oks, self.wall_ms_per_person, self.nvml_peak_bytes)


def pareto_front(points: dict[CaseKey, Point]) -> set[CaseKey]:
    def dominates(reference: Point, candidate: Point) -> bool:
        pairs = list(zip(reference.costs(), candidate.costs(), strict=True))
        return all(a <= b for a, b in pairs) and any(a < b for a, b in pairs)

    return {
        key
        for key, point in points.items()
        if not any(
            other_key[2] == key[2] and dominates(other, point)
            for other_key, other in points.items()
        )
    }


def _reduction(candidate: float, reference: float) -> float:
    return 100 * (reference - candidate) / reference


def improvements(
    row: dict[str, Any], reference: dict[str, Any]
) -> dict[str, float | None]:
    oks = 100 * (row["mean_oks"] - reference["mean_oks"])
    pck = 100 * (row["pck_0_10"] - reference["pck_0_10"])
    valid_speed = row["speed_status"] == reference["speed_status"] == "ok"
    has_memory = (
        row["nvml_peak_mib"] is not None and reference["nvml_peak_mib"] is not None
    )
    return {
        "oks_delta_pp": oks,
        "pck_0_10_delta_pp": pck,
        "p95_px_reduction_pct": _reduction(row["p95_px"], reference["p95_px"]),
        "wall_p50_reduction_pct": _reduction(
            row["wall_p50_ms_person"], reference["wall_p50_ms_person"]
        )
        if valid_speed
        else None,
        "nvml_peak_reduction_pct": _reduction(
            row["nvml_peak_mib"], reference["nvml_peak_mib"]
        )
        if valid_speed and has_memory
        else None,
        "throughput_gain_pct": 100
        * (row["persons_per_s"] / reference["persons_per_s"] - 1)
        if valid_speed
        else None,
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_bytes())


def validate_samples(
    samples: Sequence[dict[str, Any]],
    schedule: TimingSchedule,
    batch: int,
    *,
    complete: bool,
) -> None:
    protocol = schedule.protocol
    expected_count = protocol.blocks * protocol.iterations_per_block
    if len(samples) > expected_count or (complete and len(samples) != expected_count):
        raise ValueError("Report has a different count of timed iterations")
    for index, sample in enumerate(samples):
        if (
            tuple(sample["ids"]) != schedule.batch_ids[index % len(schedule.batch_ids)]
            or sample["items"] != batch
            or sample["processed_items"] != batch
            or sample["block"] != 1 + index // protocol.iterations_per_block
        ):
            raise ValueError(
                f"Report departed from the pinned timing schedule at sample {index}"
            )


def _assets(record: dict[str, Any]) -> tuple[set[str], set[str]]:
    return (
        {a["sha256"] for a in record["weights"].values()},
        {a["sha256"] for a in record["engines"].values()},
    )


def _accuracy() -> tuple[
    dict[tuple[str, str], dict[str, str]], dict[tuple[str, str], Any]
]:
    records = {}
    for family in ("vitpose", "rtmpose", "pmpose"):
        doc = _load_json(RUN / f"{family}_accuracy.json")
        if doc["bundle_sha256"] != ACCURACY_SHA256:
            raise ValueError(f"Unexpected accuracy manifest for {family}")
        for model, cases in doc["conditions"].items():
            for backend, record in cases.items():
                if model in MODELS and backend in BACKENDS:
                    records[model, backend] = record
    with (RUN / "accuracy.csv").open(encoding="utf-8", newline="") as handle:
        selected = [
            r
            for r in csv.DictReader(handle)
            if r["comparison"] == "primary" and r["backend"] in BACKENDS
        ]
    scores = {(r["model"], r["backend"]): r for r in selected}
    if (
        len(scores) != len(selected)
        or set(scores) != set(records)
        or set(scores) != set(product(MODELS, BACKENDS))
    ):
        raise ValueError("Expected all primary accuracy cases exactly once")
    for key, row in scores.items():
        record = records[key]
        if row["status"] != "ok" or row["evidence"] != record["job_id"]:
            raise ValueError(f"Missing pinned accuracy for {key}")
        if "checkpoint" in record:
            trt = record.get("tensorrt")
            records[key] = {
                "weights": {"checkpoint": record["checkpoint"]},
                "engines": {"plan": trt["plan"]} if trt else {},
            }
    if len({(r["rows"], r["eligible_joints"]) for r in scores.values()}) != 1:
        raise ValueError("Accuracy rows have different fixed denominators")
    return scores, records


def _baseline() -> dict[CaseKey, tuple[dict[str, Any], Path, str]]:
    doc = _load_json(BASELINE)
    cases = {}
    for record in doc["conditions"]:
        case = record["case"]
        if case["name"] != "vitpose-flip":
            continue
        key = REFERENCE_MODEL, case["backend"], case["batch_size"]
        if key in cases or record["process_exit_code"] != 0:
            raise ValueError(f"Invalid reused ViTPose-H speed record for {key}")
        cases[key] = (record, BASELINE, record["report_sha256"])
    if set(cases) != set(product((REFERENCE_MODEL,), BACKENDS, BATCH_SIZES)):
        raise ValueError("The ViTPose-H baseline is incomplete")
    return cases


def _check_harness(
    key: CaseKey, record: dict[str, Any], fixed: PinnedBenchmarkInputs
) -> None:
    case = record["case"]
    settings = case["settings"]
    schedule = fixed.schedule("crops", key[2])
    if (
        case["item_unit"] != "person-crop"
        or case["input_manifest_sha256"] != fixed.manifest_sha256
        or settings["dataset_id"] != INPUT_ID
        or settings["timing_protocol_sha256"] != fixed.timing_sha256
        or settings["timed_input_ids_sha256"] != schedule.timed_ids_sha256
        or record["protocol"]
        != {
            "batch_sizes": list(schedule.protocol.batch_sizes),
            "warmup": schedule.protocol.warmup,
            "iterations_per_block": schedule.protocol.iterations_per_block,
            "blocks": schedule.protocol.blocks,
        }
    ):
        raise ValueError(f"Input or timing protocol mismatch for {key}")
    if (
        record["parity_status"] != "not_evaluated"
        or record["parity"] is not None
        or "PT32計測はスコープ外" not in (record["parity_not_evaluated_reason"] or "")
    ):
        raise ValueError(f"Missing the explicit parity opt-out for {key}")
    if key[0] != REFERENCE_MODEL:
        if settings["harness_merge_commit"] != HARNESS_MERGE:
            raise ValueError(f"Wrong harness revision for {key}")
        validate_samples(
            record["samples"], schedule, key[2], complete=record["status"] == "ok"
        )


def _speed_rows(
    fixed: PinnedBenchmarkInputs,
    reports: Sequence[Path],
    omissions: Sequence[dict[str, Any]],
) -> dict[CaseKey, dict[str, Any]]:
    cases = _baseline()
    omitted = {}
    for entry in omissions:
        key = entry["model"], entry["backend"], entry["batch_size"]
        if (
            key in cases
            or key in omitted
            or not entry["reason"].strip()
            or not entry["evidence"].strip()
        ):
            raise ValueError(f"Invalid or duplicate omitted speed case {key}")
        omitted[key] = entry
    for path in reports:
        record = _load_json(path)
        case = record["case"]
        key = case["name"], case["backend"], case["batch_size"]
        if key in cases or key in omitted:
            raise ValueError(f"Duplicate speed evidence for {key}")
        cases[key] = (record, path, dual_sha256(path))

    expected = set(product(MODELS, BACKENDS, BATCH_SIZES))
    available = set(cases) | set(omitted)
    if available != expected:
        raise ValueError(
            f"Missing speed evidence for {sorted(expected - available)}. "
            f"Unexpected cases {sorted(available - expected)}."
        )

    baseline = cases[REFERENCE_MODEL, "PT16", 1][0]["environment"]
    output = {}
    for key in sorted(expected):
        omission = omitted.get(key)
        if omission is not None:
            output[key] = _speed_summary(
                "infeasible", omission["reason"], omission["evidence"]
            )
            continue
        record, path, digest = cases[key]
        _check_harness(key, record, fixed)
        status = record["status"]
        reason = None
        if status == "infeasible" and record["summary"] is None and record["failure"]:
            failure = record["failure"]
            reason = f"{failure['phase']}: {failure['message']}"
        elif status != "ok" or record["failure"] is not None:
            raise ValueError(f"Unexpected speed status for {key}: {status}")
        if status == "ok" and any(
            record["environment"].get(k) != baseline[k] for k in DEVICE_KEYS
        ):
            raise ValueError(
                f"The accelerator or runtime differs from ViTPose-H for {key}"
            )
        output[key] = _speed_summary(status, reason, str(path), record, digest)
    return output


def _speed_summary(
    status: str,
    reason: str | None,
    source: str,
    record: dict[str, Any] | None = None,
    digest: str | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    commit = ""
    if record is not None:
        settings = record["case"]["settings"]
        commit = settings.get("execution_commit", settings.get("source_commit"))
        if status == "ok":
            summary = record["summary"]
            nvml_peak = record["nvml_device_peak_sampled_bytes"]
            metrics = {
                "wall_p50_ms_person": summary["stage_wall"]["p50_ms_per_item"],
                "wall_p95_ms_person": summary["stage_wall"]["p95_ms_per_item"],
                "gpu_p50_ms_person": summary["model_gpu"]["p50_ms_per_item"],
                "gpu_p95_ms_person": summary["model_gpu"]["p95_ms_per_item"],
                "persons_per_s": summary["stage_wall"]["items_per_second"],
                "nvml_peak_mib": nvml_peak / 2**20 if nvml_peak is not None else None,
                "torch_allocated_mib": record["torch_peak_allocated_bytes"] / 2**20,
                "torch_reserved_mib": record["torch_peak_reserved_bytes"] / 2**20,
            }
            if not all(
                math.isfinite(v) and v >= 0 for v in metrics.values() if v is not None
            ):
                raise ValueError(f"Invalid speed or memory values in {source}")
            if min(metrics["wall_p50_ms_person"], metrics["persons_per_s"]) <= 0:
                raise ValueError(
                    f"Expected positive wall time and throughput in {source}"
                )
    return {
        "speed_status": status,
        "speed_reason": reason,
        "parity": PARITY_LABEL,
        "execution_commit": commit,
        "speed_source": source,
        "speed_report_sha256": digest,
        "record": record,
        **dict.fromkeys(
            (
                "wall_p50_ms_person",
                "wall_p95_ms_person",
                "gpu_p50_ms_person",
                "gpu_p95_ms_person",
                "persons_per_s",
                "nvml_peak_mib",
                "torch_allocated_mib",
                "torch_reserved_mib",
            )
        ),
        **metrics,
    }


def _compare(
    speeds: dict[CaseKey, dict[str, Any]],
    scores: dict[tuple[str, str], dict[str, str]],
    accuracy_records: dict[tuple[str, str], Any],
) -> list[dict[str, Any]]:
    rows = {}
    points = {}
    for key, speed in speeds.items():
        model, backend, batch = key
        score = scores[model, backend]
        record = speed["record"]
        row = {
            "model": model,
            "backend": backend,
            "batch_size": batch,
            "accuracy_rows": int(score["rows"]),
            "eligible_joints": int(score["eligible_joints"]),
            "missing_joints": int(score["missing_joints"]),
            "mean_oks": float(score["oks"]),
            "pck_0_10": float(score["pck_0_10"]),
            "pck_0_20": float(score["pck_0_20"]),
            "p95_px": float(score["p95_px"]),
            "accuracy_job": score["evidence"],
            **{name: value for name, value in speed.items() if name != "record"},
        }
        if row["speed_status"] == "ok":
            if _assets(record) != _assets(accuracy_records[model, backend]):
                raise ValueError(f"Accuracy and speed model assets differ for {key}")
            if row["nvml_peak_mib"] is not None:
                points[key] = Point(
                    row["mean_oks"],
                    row["wall_p50_ms_person"],
                    record["nvml_device_peak_sampled_bytes"],
                )
        rows[key] = row
    front = pareto_front(points)
    return [
        {
            **row,
            "pareto": key in front if key in points else None,
            **improvements(row, rows[REFERENCE_MODEL, key[1], key[2]]),
        }
        for key, row in rows.items()
    ]


def _plot(rows: Sequence[dict[str, Any]], output: Path) -> None:
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, 4.2), layout="constrained")
    axes = figure.subplots(1, 2, sharey=True)
    colors = dict(
        zip(
            MODELS, ("#38518A", "#148F77", "#58B496", "#AE632D", "#D75165"), strict=True
        )
    )
    for row in rows:
        if row["batch_size"] != 1 or row["speed_status"] != "ok":
            continue
        for ax, metric in zip(
            axes, ("wall_p50_ms_person", "nvml_peak_mib"), strict=True
        ):
            if row[metric] is None:
                continue
            ax.scatter(
                row[metric],
                row["mean_oks"],
                s=85,
                marker="o" if row["backend"] == "PT16" else "s",
                color=colors[row["model"]],
                edgecolor="#111827" if row["pareto"] else "none",
                linewidth=1.55,
                label=f"{row['model']} {row['backend']}",
            )
    axes[0].set_xlabel("Wall P50 (ms/person)")
    axes[1].set_xlabel("Sampled NVML device peak (MiB)")
    axes[0].set_ylabel("Fixed-box mean OKS")
    for ax in axes:
        ax.grid(alpha=0.22)
    figure.suptitle(
        "Meiji 4cam pose at batch 1 · outline marks the three-metric Pareto frontier"
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=5, fontsize=8)
    figure.savefig(output, dpi=150, facecolor="white")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--omissions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("reports", nargs="*", type=Path)
    args = parser.parse_args()
    fixed = PinnedBenchmarkInputs(
        args.inputs,
        TIMING_PROTOCOL,
        manifest_sha256=INPUT_MANIFEST_SHA256,
        timing_sha256=TIMING_PROTOCOL_SHA256,
    )
    scores, records = _accuracy()
    speeds = _speed_rows(
        fixed, args.reports, _load_json(args.omissions) if args.omissions else ()
    )
    rows = _compare(speeds, scores, records)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "comparison.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, args.output / "pareto-batch1.png")


if __name__ == "__main__":
    main()
