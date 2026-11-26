"""Verify the FastHMR timing reports and summarize them beside the frozen baseline."""

import hashlib
import json
import math
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from statistics import median, quantiles

BATCHES = (1, 2, 4)
WINDOW = 16
WARMUP, BLOCKS, REPS = 30, 3, 100
RUN_HEAD = "8a25fe12b543656cb1e883e7251bd1a3fd299fd0"
HARNESS = "13e508cb78c46663fbe58ad8be03bcf9c96816a6"
INPUT_SHA = "64c74d75e5f1afe869911f1620b9c8e8a793fc4c4e4ac830f4abd97b21fb1a62"
TIMING_SHA = "39e362cb87c7c373f8c1ad25e5e28025aab89b1467be21ca3139c2fb456136a2"
PARITY_REASON = "未判定（PT32計測はスコープ外、ユーザー判断 2026-11-25）"
FORWARDS_PER_SEQUENCE = {
    "yolo11x_seg": 4,
    "humanfov": 4,
    "camerahmr_eclm_mask_tome": 8,
    "fast_hmr_diffusion_vae": 2,
}
VRAM_FIELDS = (
    "torch_peak_allocated_bytes",
    "torch_peak_reserved_bytes",
    "nvml_device_peak_sampled_bytes",
    "nvml_peak_minus_before_load_bytes",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def read_pinned(path, expected):
    data = path.read_bytes()
    require(digest(data) == expected, f"Pinned file differs: {path}")
    return json.loads(data)


def input_ids(bundle):
    directory = bundle.with_name("run-person-model-speed-baseline")
    manifest = read_pinned(directory / "inputs/manifest.json", INPUT_SHA)
    schedule = read_pinned(directory / "timing-protocol.json", TIMING_SHA)
    index = manifest["indexes"]["body_sequences"]
    rows = read_pinned(directory / "inputs" / index["path"], index["sha256"])
    ids = tuple(row["id"] for row in rows)
    require(len(ids) == len(set(ids)) == 24, "Fixed work requires 24 unique series")
    return ids, schedule["schedules"]["body_sequences"]["batch_sizes"]


def check_stats(summary, durations, items):
    p50 = median(durations)
    p95 = quantiles(durations, n=100, method="inclusive")[94]
    expect = {
        "samples": len(durations),
        "items": items * len(durations),
        "p50_ms": p50,
        "p95_ms": p95,
        "p50_ms_per_item": p50 / items,
        "p95_ms_per_item": p95 / items,
        "total_ms": sum(durations),
        "items_per_second": 1000 * items * len(durations) / sum(durations),
    }
    for key, value in expect.items():
        require(
            math.isclose(summary[key], value, rel_tol=1e-9),
            f"Summary disagrees on {key}",
        )


def model_gpu_times(sample, size):
    by_model = Counter()
    totals = Counter()
    for stage in sample["stages"]:
        model, gpu, kind = stage["name"], stage["gpu_ms"], stage["stage"]
        if kind == "forward":
            require(gpu > 0 and math.isfinite(gpu), "Invalid forward CUDA time")
            by_model[model] += 1
            totals[model] += gpu
    require(
        by_model == {k: count * size for k, count in FORWARDS_PER_SEQUENCE.items()},
        "Timed forward calls differ from the fixed FastHMR workload",
    )
    return totals


def check_case(path, size, ids, schedule_sha):
    received = path.read_bytes()
    report = json.loads(received)
    case, protocol = report["case"], report["protocol"]
    settings = case["settings"]
    require(
        report["status"] == "ok"
        and all(report[key] is None for key in ("parity", "failure")),
        f"Missing successful timing result: {path}",
    )
    require(
        report["parity_status"] == "not_evaluated"
        and report["parity_not_evaluated_reason"] == PARITY_REASON,
        f"Unexpected parity mode: {path}",
    )
    require(
        case["name"] == "fasthmr-camerahmr"
        and case["backend"] == "PT16"
        and case["batch_size"] == size
        and case["item_unit"] == "person-frame"
        and case["input_manifest_sha256"] == INPUT_SHA
        and settings["timing_protocol_sha256"] == TIMING_SHA
        and settings["timed_input_ids_sha256"] == schedule_sha
        and settings["source_commit"] == RUN_HEAD
        and settings["harness_commit"] == HARNESS
        and settings["body_window_frames"] == WINDOW
        and not settings["padding"]
        and settings["feature_batch_size"] == 4
        and settings["sequence_execution"] == "serial",
        f"Unexpected timing case: {path}",
    )
    require(
        protocol
        == {
            "batch_sizes": list(BATCHES),
            "blocks": BLOCKS,
            "iterations_per_block": REPS,
            "warmup": WARMUP,
        },
        "Unexpected warmup or timing blocks",
    )
    require(
        "5060 Ti" in report["environment"]["gpu"] and not report["engines"],
        "Unexpected GPU or model backend",
    )

    fixed_batches = [ids[i : i + size] for i in range(0, len(ids), size)]
    samples = report["samples"]
    require(len(samples) == BLOCKS * REPS, "Expected 300 timed repetitions")
    expected_items = size * WINDOW
    gpu_times, walls, model_times = (
        [],
        [],
        {model: [] for model in FORWARDS_PER_SEQUENCE},
    )
    for i, sample in enumerate(samples):
        require(
            tuple(sample["ids"]) == fixed_batches[i % len(fixed_batches)]
            and sample["block"] == 1 + i // REPS
            and sample["items"] == sample["processed_items"] == expected_items,
            f"Unexpected input order or item count at B={size}, repetition={i + 1}",
        )
        totals = model_gpu_times(sample, size)
        for model, time in totals.items():
            model_times[model].append(time)
        gpu_time, wall_time = sum(totals.values()), sample["wall_ms"]
        require(math.isfinite(wall_time) and wall_time >= gpu_time, "Invalid wall time")
        gpu_times.append(gpu_time)
        walls.append(wall_time)

    summary = report["summary"]
    check_stats(summary["model_gpu"], gpu_times, expected_items)
    check_stats(summary["stage_wall"], walls, expected_items)
    vram = {key: report[key] for key in VRAM_FIELDS}
    nvml = report["nvml_used_bytes"]
    require(
        vram["nvml_device_peak_sampled_bytes"] == max(nvml.values()),
        "NVML summary differs from the sampled maximum",
    )
    return {
        "case": case,
        "status": report["status"],
        "parity_status": report["parity_status"],
        "environment": report["environment"],
        "setup_wall_ms": report["setup_wall_ms"],
        "summary": summary,
        "component_gpu_p50_ms": {
            key: median(values) for key, values in model_times.items()
        },
        **vram,
        "raw_report": {
            "file": path.name,
            "bytes": len(received),
            "sha256": digest(received),
        },
    }, report["weights"]


def verify(directory, bundle):
    source_sha = {}
    for line in (bundle / "source.sha256.txt").read_text().splitlines():
        sha, captured = line.split(maxsplit=1)
        name = Path(captured).name
        require(
            digest((bundle / name).read_bytes()) == sha,
            f"Captured source differs: {name}",
        )
        source_sha[name] = sha
    require(
        set(source_sha) == {"benchmark_fasthmr.py", "run.sh"},
        "Incomplete source receipt",
    )
    ids, schedule = input_ids(bundle)
    baseline_file = (
        bundle.with_name("run-person-model-speed-low-precision") / "cases.json"
    )
    baseline = json.loads(baseline_file.read_text())
    require(
        baseline["measurement_commit"] == HARNESS
        and baseline["input_manifest_sha256"] == INPUT_SHA,
        "Final #2497 baseline differs from the recorded measurement protocol",
    )
    bodies = {
        item["case"]["batch_size"]: item
        for item in baseline["conditions"]
        if item["case"]["name"] == "body" and item["case"]["backend"] == "PT16"
    }
    cases, asset_set = [], None
    for size in BATCHES:
        summary, assets = check_case(
            directory / f"pt16-b{size}.json", size, ids, schedule[str(size)]
        )
        if asset_set is None:
            asset_set = assets
        require(assets == asset_set, "FastHMR asset hashes differ between batch sizes")
        current = bodies[size]
        require(
            current["status"] == "ok"
            and current["parity_status"] == "not_evaluated"
            and current["case"]["settings"]["timed_input_ids_sha256"]
            == schedule[str(size)]
            and current["case"]["settings"]["timing_protocol_sha256"] == TIMING_SHA
            and current["case"]["settings"]["body_window_frames"] == WINDOW
            and current["case"]["input_manifest_sha256"] == INPUT_SHA
            and current["protocol"]
            == {
                "batch_sizes": list(BATCHES),
                "blocks": BLOCKS,
                "iterations_per_block": REPS,
                "warmup": WARMUP,
            }
            and current["samples"] == BLOCKS * REPS,
            "GVHMR and FastHMR workloads differ",
        )
        for field in (
            "gpu",
            "driver",
            "torch",
            "cuda",
            "power_limit_mw",
            "fp32_precision",
        ):
            require(
                summary["environment"][field] == current["environment"][field],
                f"Environment mismatch for {field}",
            )
        stage = current["summary"]["stage_wall"]
        summary["gvhmr_pt16"] = {
            "case": current["case"],
            "stage_wall": stage,
            "model_gpu": current["summary"]["model_gpu"],
            **{key: current[key] for key in VRAM_FIELDS},
        }
        summary["stage_wall_p50_ratio_to_gvhmr"] = (
            summary["summary"]["stage_wall"]["p50_ms"] / stage["p50_ms"]
        )
        cases.append(summary)

    return {
        "harness_commit": HARNESS,
        "source_commit": RUN_HEAD,
        "input_manifest_sha256": INPUT_SHA,
        "timing_protocol_sha256": TIMING_SHA,
        "baseline_case_file_sha256": digest(baseline_file.read_bytes()),
        "fixed_body_sequences": len(ids),
        "parity_not_evaluated_reason": PARITY_REASON,
        "source_sha256": source_sha,
        "weights": {
            key: {k: value[k] for k in ("bytes", "sha256")}
            for key, value in asset_set.items()
        },
        "cases": cases,
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            verify(args.predictions, Path(__file__).resolve().parent),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
