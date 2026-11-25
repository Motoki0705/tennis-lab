"""Verify completed queue output and preserve the benchmark evidence in knowledge."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from math import isclose, isfinite
from pathlib import Path
from statistics import median, quantiles
from tempfile import TemporaryDirectory

import numpy as np
from run import CASE_UNITS, GROUPS, SWEEPS
from speed_inputs import BODY_LENGTH, INPUT_ID, require_sha

from src.utils.checksum import dual_sha256
from src.utils.io import write_json_atomic

BACKENDS = ("PT32", "PT16", "TRT16")
STAGES = ("preprocess", "h2d", "forward", "postprocess", "d2h")
RAW_FIELDS = (
    "batch_size",
    "iteration",
    "block",
    "batch_index",
    "items",
    "processed_items",
    "stage_wall_ms",
    "model_gpu_ms",
    *(f"{name}_ms" for name in STAGES),
)
CASE_FIELDS = (
    "workload",
    "backend",
    "batch_size",
    "item_unit",
    "status",
    "samples",
    "distinct_inputs",
    "timed_clips",
    "gpu_p50_ms",
    "gpu_p95_ms",
    "wall_p50_ms",
    "wall_p95_ms",
    "wall_items_per_second",
    "torch_peak_allocated_bytes",
    "torch_peak_reserved_bytes",
    "nvml_device_peak_sampled_bytes",
    "nvml_peak_minus_before_load_bytes",
    "load_ms",
    "cold_start_ms",
)


def read_json(path: Path):
    return json.loads(path.read_bytes())


def pinned_inputs(source: Path, destination: Path):
    directory = source / "inputs"
    manifest_path = directory / "manifest.json"
    manifest = read_json(manifest_path)
    digest = dual_sha256(manifest_path)
    if manifest["id"] != INPUT_ID:
        raise ValueError(f"Wrong input collection: {manifest['id']}")
    exports = {manifest_path}
    indexes, verified = {}, {}
    for name, expected in manifest["indexes"].items():
        index_path = require_sha(directory / expected["path"], expected["sha256"])
        rows = read_json(index_path)
        ids = [row["id"] for row in rows]
        if len(set(ids)) != expected["records"] or len(ids) != expected["records"]:
            raise ValueError(f"Wrong record count or duplicate ID: {index_path}")
        for row in rows:
            relative, expected_hash = row["path"], row["sha256"]
            if relative in verified and verified[relative] != expected_hash:
                raise ValueError(f"Duplicate path with conflicting hashes: {relative}")
            if relative not in verified:
                require_sha(directory / relative, expected_hash)
                verified[relative] = expected_hash
        indexes[name] = ids
        exports.add(index_path)

    keep = manifest["pose_sweep_crop_count"]
    if indexes["crops"][keep:] != manifest["pose_sweep_tail_ids"]:
        raise ValueError("Crop sweep and tail IDs disagree")
    for path in sorted(exports):
        (destination / "inputs").mkdir(exist_ok=True)
        (destination / "inputs" / path.name).write_bytes(path.read_bytes())
    print(f"Verified {len(verified)} fixed input files and manifest {digest}")
    return (
        digest,
        {
            "dino": indexes["frames"],
            "dino-vitpose": indexes["frames"],
            "vitpose": indexes["crops"][:keep],
            "vitpose-flip": indexes["crops"][:keep],
            "body": indexes["body_sequences"],
        },
        len(verified),
    )


def sample_stats(pairs):
    values = [ms for ms, _ in pairs]
    per_item = [ms / n for ms, n in pairs]
    if not all(isfinite(v) and v > 0 for v in values):
        raise ValueError("Found a nonpositive or nonfinite timing sample")

    def p95(ms):
        return quantiles(ms, n=20, method="inclusive")[-1] if len(ms) > 1 else ms[0]

    return {
        "samples": len(values),
        "items": sum(n for _, n in pairs),
        "total_ms": sum(values),
        "p50_ms": median(values),
        "p95_ms": p95(values),
        "p50_ms_per_item": median(per_item),
        "p95_ms_per_item": p95(per_item),
        "items_per_second": 1000 * sum(n for _, n in pairs) / sum(values),
    }


def verify_samples(report, pinned_ids, csv_writer):
    case, protocol = report["case"], report["protocol"]
    size = case["batch_size"]
    if protocol["batch_sizes"] != list(SWEEPS[case["name"]]) or (
        protocol["warmup"],
        protocol["iterations_per_block"],
        protocol["blocks"],
    ) != (30, 100, 3):
        raise ValueError(f"Benchmark protocol changed: {case}")
    ids = [pinned_ids[i : i + size] for i in range(0, len(pinned_ids), size)]
    if any(len(row) != size for row in ids) or len(report["samples"]) != 300:
        raise ValueError(f"Missing full batches or expected repetitions: {case}")
    timings, forwards, covered = defaultdict(list), defaultdict(list), set()
    expected_items = size * (BODY_LENGTH if case["name"] == "body" else 1)
    for index, sample in enumerate(report["samples"]):
        block = index // protocol["iterations_per_block"] + 1
        batch = index % len(ids)
        if (
            sample["ids"] != ids[batch]
            or sample["block"] != block
            or (sample["items"], sample["processed_items"])
            != (expected_items, expected_items)
        ):
            raise ValueError(
                f"Input address or repetition changed: {case}, sample {index}"
            )
        covered.update(sample["ids"])
        wall_by_stage, gpu = defaultdict(float), 0.0
        for stage in sample["stages"]:
            wall_by_stage[stage["stage"]] += stage["wall_ms"]
            if stage["gpu_ms"] is not None:
                if stage["stage"] != "forward":
                    raise ValueError(f"Unexpected CUDA event outside forward: {stage}")
                gpu += stage["gpu_ms"]
                forwards[stage["name"]].append((stage["gpu_ms"], stage["wall_ms"]))
        if not wall_by_stage.keys() <= set(STAGES):
            raise ValueError(f"Unknown measurement stages: {wall_by_stage.keys()}")
        wall = sample["wall_ms"]
        for name, ms in {"stage_wall": wall, "model_gpu": gpu, **wall_by_stage}.items():
            timings[name].append((ms, expected_items))
        csv_writer.writerow(
            (
                size,
                index + 1,
                block,
                batch,
                expected_items,
                sample["processed_items"],
                wall,
                gpu,
                *(wall_by_stage.get(name) for name in STAGES),
            )
        )
    summary = {name: sample_stats(values) for name, values in timings.items()}
    official = report["summary"]
    if report["status"] == "ok":
        if official.keys() != summary.keys() or any(
            not isclose(value, official[group][metric], rel_tol=1e-11)
            for group, metrics in summary.items()
            for metric, value in metrics.items()
        ):
            raise ValueError(f"Raw samples disagree with the accepted summary: {case}")
    elif official is not None:
        raise ValueError(f"A failed case exposes accepted timing statistics: {case}")
    return {
        "distinct_inputs": len(covered),
        "timed_clips": sorted({"/".join(ident.split("/")[:2]) for ident in covered}),
        "forward_medians_ms": {
            name: {
                "gpu": median(gpu for gpu, _ in values),
                "wall": median(wall for _, wall in values),
            }
            for name, values in forwards.items()
        },
    }


def archive_asset_paths(assets, source):
    for name, asset in assets.items():
        path = asset["path"]
        if "/ckpt/" in path:
            asset["path"] = "ckpt/" + path.split("/ckpt/", 1)[1]
        elif "/tensorrt/" in path:
            asset["path"] = "${PERSON_TRT_ROOT}/" + path.split("/tensorrt/", 1)[1]
        elif Path(path).resolve().is_relative_to(source.resolve()):
            asset["path"] = "${PERSON_RUN_OUTPUT}/" + str(
                Path(path).resolve().relative_to(source.resolve())
            )
        elif name == "dino_msda_cuda":
            asset["path"] = "${TENNIS_LAB_DINO_EXTENSION}"
        else:
            raise ValueError(f"Unknown weight or engine asset path: {path}")


def pelvis_aligned_joint_error_p95(source: Path) -> float:
    reference = read_json(source / "results/body/PT32-b1/predictions.json")
    candidate = read_json(source / "results/body/TRT16-b1/predictions.json")
    if reference.keys() != candidate.keys():
        raise ValueError("Body prediction IDs differ")
    distances = []
    for key, value in reference.items():
        first, second = (
            np.asarray(value["joints"]),
            np.asarray(candidate[key]["joints"]),
        )
        distances.append(
            np.linalg.norm(
                (first - first[:, :1]) - (second - second[:, :1]), axis=-1
            ).ravel()
        )
    return 1000 * float(np.percentile(np.concatenate(distances), 95))


def record(source: Path, destination: Path):
    digest, ids_by_workload, file_count = pinned_inputs(source, destination)
    defined = {
        (workload, backend, size)
        for workload, sweep in SWEEPS.items()
        for backend in BACKENDS
        for size in sweep
    }
    attempts = {}
    for group, names in GROUPS.items():
        for row in read_json(source / f"attempts-{group}.json"):
            key = row["workload"], row["backend"], row["batch"]
            if key in attempts or row["workload"] not in names or row["exit_code"] != 0:
                raise ValueError(f"Unresolved, duplicate or wrong-group attempt: {row}")
            attempts[key] = row
    if attempts.keys() != defined:
        raise ValueError("Queue attempt records do not cover the frozen case matrix")

    (destination / "measurements").mkdir(exist_ok=True)
    cases, summaries = [], []
    for workload, sweep in SWEEPS.items():
        for backend in BACKENDS:
            relative_csv = f"measurements/{workload}-{backend.lower()}.csv"
            with (destination / relative_csv).open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.writer(file, lineterminator="\n")
                writer.writerow(RAW_FIELDS)
                for size in sweep:
                    attempt = attempts[workload, backend, size]
                    path = source / attempt["path"] / "report.json"
                    report = read_json(path)
                    case = report["case"]
                    if (
                        case["name"],
                        case["backend"],
                        case["batch_size"],
                        case["item_unit"],
                        case["input_manifest_sha256"],
                    ) != (
                        workload,
                        backend,
                        size,
                        CASE_UNITS[workload],
                        digest,
                    ) or report["status"] != attempt["status"]:
                        raise ValueError(
                            f"Report disagrees with frozen input, attempt or condition: {path}"
                        )
                    if report["status"] != (
                        "ok" if report["parity"]["passed"] else "parity_failed"
                    ):
                        raise ValueError(f"Mismatched parity outcome: {path}")
                    if report["parity"]["metrics"]["checked_ids"] != len(
                        ids_by_workload[workload]
                    ):
                        raise ValueError(f"Incomplete parity input coverage: {path}")
                    coverage = verify_samples(report, ids_by_workload[workload], writer)
                    count = len(report.pop("samples"))
                    accepted = report["summary"] or {}
                    gpu, wall = (
                        accepted.get("model_gpu", {}),
                        accepted.get("stage_wall", {}),
                    )
                    row = {
                        "workload": workload,
                        "backend": backend,
                        "batch_size": size,
                        "item_unit": case["item_unit"],
                        "status": report["status"],
                        "samples": count,
                        "distinct_inputs": coverage["distinct_inputs"],
                        "timed_clips": len(coverage["timed_clips"]),
                        "gpu_p50_ms": gpu.get("p50_ms"),
                        "gpu_p95_ms": gpu.get("p95_ms"),
                        "wall_p50_ms": wall.get("p50_ms"),
                        "wall_p95_ms": wall.get("p95_ms"),
                        "wall_items_per_second": wall.get("items_per_second"),
                        **{key: report[key] for key in CASE_FIELDS if "bytes" in key},
                        "load_ms": report["setup_wall_ms"]["load"],
                        "cold_start_ms": report["setup_wall_ms"]["cold_start"],
                    }
                    for group in ("weights", "engines"):
                        archive_asset_paths(report[group], source)
                    cases.append(
                        {
                            **report,
                            **coverage,
                            "samples": count,
                            "raw_csv": relative_csv,
                            "report_sha256": dual_sha256(path),
                            "case_log_sha256": dual_sha256(path.with_name("case.log")),
                            "process_exit_code": attempt["exit_code"],
                        }
                    )
                    summaries.append(row)
    commits = {item["case"]["settings"]["source_commit"] for item in cases}
    if len(commits) != 1:
        raise ValueError(f"Expected one measured code revision, got {commits}")
    with (destination / "cases.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, CASE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)
    write_json_atomic(
        destination / "cases.json",
        {
            "measurement_commit": commits.pop(),
            "input_manifest_sha256": digest,
            "fixed_input_files_verified": file_count,
            "conditions": cases,
            "diagnostics": {
                "body_trt16_b1_pelvis_aligned_joint_error_p95_mm": pelvis_aligned_joint_error_p95(
                    source
                )
            },
        },
    )
    build = source / "vitpose-b1-build.json"
    (destination / build.name).write_bytes(build.read_bytes())
    counts = Counter(row["status"] for row in summaries)
    print(
        f"Verified {len(cases)} conditions, {sum(row['samples'] for row in summaries)} raw repetitions: {dict(counts)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=".person-speed-record-", dir=args.destination.parent
    ) as staging:
        staging = Path(staging)
        record(args.source, staging)
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                target = args.destination / path.relative_to(staging)
                target.parent.mkdir(parents=True, exist_ok=True)
                path.replace(target)
