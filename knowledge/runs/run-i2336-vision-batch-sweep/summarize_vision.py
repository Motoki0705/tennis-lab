"""Verify and publish the timing, accuracy and profiler evidence from #2336."""

import argparse
import math
from pathlib import Path

from benchmark import write_json
from convert import TRT_PRECISIONS, verify_compute_precision
from convert_dino import verify_layers
from inputs import sha256_file
from profiles import (
    ncu_launch_shapes,
    ncu_rows,
    nsys_launch_shapes,
    read_ncu,
    read_nsys,
)
from runtime import BLOCKS, REPEATS_PER_BLOCK, WARMUP, stats
from sweep import case_id
from vision_inputs import (
    BATCHES,
    CONDITIONS,
    DINO,
    HMR2,
    OPT_BATCH,
    VITPOSE,
    PinnedInputs,
    load_conversion,
    read,
)

RUNS = Path(__file__).resolve().parents[1]
LABELS = {VITPOSE: "vit", HMR2: "hmr2", DINO: "dino"}
MEMORY_KEYS = (
    "torch_peak_allocated_bytes",
    "torch_peak_reserved_bytes",
    "nvml_device_peak_sampled_bytes",
    "nvml_device_peak_minus_idle_bytes",
)
DEVICE_KEYS = ("device", "device_uuid", "driver", "torch", "cuda", "cc")


def within_draft(model, accuracy):
    if model == VITPOSE:
        xy, confidence = accuracy["xy_l2_px"], accuracy["confidence_abs"]
        return bool(
            accuracy["mask_mismatches"] == 0
            and xy
            and confidence
            and xy["p95"] <= 1
            and xy["max"] <= 5
            and confidence["p95"] <= 0.02
        )
    if model == HMR2:
        return (
            accuracy["cosine"]["p05"] >= 0.995
            and accuracy["relative_l2"]["p95"] <= 0.05
        )
    overall = accuracy["overall"]
    return bool(
        overall["iou"]
        and overall["score_abs_error"]
        and overall["iou"]["p05"] >= 0.98
        and overall["score_abs_error"]["p95"] <= 0.01
        and overall["missing"] * 100 <= overall["reference_boxes"]
        and overall["extra"] * 100 <= overall["candidate_boxes"]
    )


def verify_timing(measurement, batch, path):
    samples = measurement["latency_samples_ms"]
    if (
        measurement["warmup"] != WARMUP
        or len(samples) != BLOCKS
        or len(measurement["blocks"]) != BLOCKS
        or any(len(block) != REPEATS_PER_BLOCK for block in samples)
        or any(not math.isfinite(ms) or ms <= 0 for block in samples for ms in block)
    ):
        raise ValueError(f"Unexpected timing sample series: {path}")
    for block, saved in zip(samples, measurement["blocks"], strict=True):
        if saved != stats(block, batch=batch):
            raise ValueError(f"Block latency does not match the raw series: {path}")
    summary = stats([ms for block in samples for ms in block], batch=batch)
    if any(measurement[key] != value for key, value in summary.items()):
        raise ValueError(f"Latency or item throughput does not match B={batch}: {path}")
    return {**summary, "warmup": WARMUP, "samples": BLOCKS * REPEATS_PER_BLOCK}


def stage_root(root, model, condition, batch):
    if condition == "1" and model == VITPOSE and batch == 32:
        return root / "retry-2/vit-1"
    if condition in {"0", "1"}:
        return root / model
    return root / "retry-2" / f"{LABELS[model]}-{condition}"


def check_plan(root, data, condition):
    stages = stage_root(root, data.model, condition, 1)
    directory = stages / f"engine-{condition}"
    plan, conversion, digest = load_conversion(data, condition, directory)
    maximum = max(BATCHES[data.model])
    profile = data.profile(OPT_BATCH[data.model], maximum)
    report = read(stages / "sweep.json")
    if (
        conversion["input_profile"] != profile
        or report["identity"] != data.identity
        or not report["stages"][f"build-{condition}"]["verified"]
        or conversion["queue_job"] != report["queue_job"]
    ):
        raise ValueError(
            f"Engine does not match its sweep or planned profile: {directory}"
        )
    precision = TRT_PRECISIONS[conversion["condition"]]
    if data.model == DINO:

        def verify(details):
            return verify_layers(
                details,
                precision,
                conversion["precision_requirements"],
                set(conversion["sampling_nodes"]),
                conversion["method"],
            )
    else:
        manifest = read(directory / "precision.json")

        def verify(details):
            return verify_compute_precision(details, precision, **manifest)

    for index, label in enumerate(("min", "opt", "max")):
        actual = verify(read(plan.with_suffix(f".{label}.layers.json")))
        if conversion["compute_precision"][label] != {
            **actual,
            "bound_input_shapes": {
                name: shapes[index] for name, shapes in profile.items()
            },
        }:
            raise ValueError(f"Inspector receipt changed at {label}: {plan}")
    rejections = {}
    for size in (0, maximum + 1):
        stage = report["stages"][f"reject-{case_id(condition, size, None)}"]
        log = stages / stage["log"]
        shape = (size * data.multiplier, *data.input_shape)
        message = f"TensorRT rejected input shape pixels={shape}"
        if (
            stage["exit_code"] <= 0
            or not stage["verified"]
            or message not in log.read_text()
        ):
            raise ValueError(f"Missing TensorRT rejection for {shape}: {log}")
        rejections[str(size)] = {
            "exit_code": stage["exit_code"],
            "message": message,
            "log": str(log.relative_to(root)),
            "log_sha256": sha256_file(log),
        }
    return verify, {
        "plan": str(plan.relative_to(root)),
        "engine_sha256": digest,
        "workspace_bytes": conversion["workspace_bytes"],
        "input_profile": profile,
        "graph_files_sha256": conversion["files"],
        "compute_precision": conversion["compute_precision"],
        "quantization": conversion["quantization"],
        "msda_method": conversion.get("method"),
        "queue_job": f"{report['queue_job']}.job",
        "out_of_range": rejections,
    }


def check_case(root, data, condition, batch, reference, reference_sha, plan_check):
    stages = stage_root(root, data.model, condition, batch)
    name = case_id(condition, batch, None)
    case = stages / name
    result, measurement = read(case / "result.json"), read(case / "measurement.json")
    report = read(stages / "sweep.json")
    nsys = read_nsys(case / "nsys.sqlite")
    if (
        not report["stages"][name]["verified"]
        or report["stages"][name]["exit_code"] != 0
        or result["queue_job"] != report["queue_job"]
        or measurement["queue_job"] != report["queue_job"]
        or result["identity"] != data.identity
        or (result["condition"], result["batch_size"]) != (condition, batch)
        or measurement["batch_shape"] != [batch, *data.input_shape]
        or result["representative_addresses"] != data.representative(batch)
        or result["prediction_sha256"] != sha256_file(case / data.prediction_name)
        or result["fp32_reference_sha256"] != reference_sha
        or result["nsys"] != {"status": "measured", **nsys}
    ):
        raise ValueError(
            f"Case receipts disagree with the fixed experiment or raw trace: {case}"
        )
    if plan_check is not None:
        verify, engine = plan_check
        if (
            result["engine_sha256"] != engine["engine_sha256"]
            or result["input_profile"] != engine["input_profile"]
            or Path(result["conversion"]).resolve() != (root / engine["plan"]).parent
            or measurement["trt_nvtx_verbosity"] != "NONE"
            or measurement["engine_batch_shape"]
            != [batch * data.multiplier, *data.input_shape]
            or verify(read(case / "layers.json"))
            != measurement["trt_compute_precision"]
        ):
            raise ValueError(f"Case used a different plan or precision: {case}")
    expected_tail = len(data.addresses) % batch
    tail = measurement.get("tail_batch")
    if expected_tail != (tail["size"] if tail else 0):
        raise ValueError(f"Tail batch differs from the fixed input order: {case}")
    timing = verify_timing(measurement, batch, case)
    if tail:
        timing["tail_batch"] = {
            "size": expected_tail,
            **verify_timing(tail, expected_tail, case),
        }
        if (
            plan_check is not None
            and verify(read(case / "tail.layers.json")) != tail["trt_compute_precision"]
        ):
            raise ValueError(
                f"Tail precision differs from the TensorRT inspector: {case}"
            )
    snapshots = measurement["nvml_device_snapshots_bytes"]
    peak = max(snapshots.values())
    if (
        measurement["nvml_device_peak_sampled_bytes"] != peak
        or measurement["nvml_device_peak_minus_idle_bytes"] != peak - snapshots["idle"]
        or not 0
        < measurement["torch_peak_allocated_bytes"]
        <= measurement["torch_peak_reserved_bytes"]
    ):
        raise ValueError(f"Peak VRAM does not match the measured case: {case}")
    accuracy = data.compare(data.predictions(case), reference)
    if accuracy != result["accuracy"]:
        raise ValueError(f"Accuracy does not match saved predictions: {case}")
    # Per-camera results and the feature-by-dimension vector remain in result.json.
    for key in (
        "by_clip_camera",
        "crop_max_xy_l2_px_by_camera",
        "relative_l2_by_camera",
        "max_abs_error_by_dimension",
    ):
        accuracy.pop(key, None)
    return (
        case,
        result,
        measurement,
        {
            "artifact": str(case.relative_to(root)),
            "queue_job": f"{result['queue_job']}.job",
            "prediction_sha256": result["prediction_sha256"],
            "timing": timing,
            "vram": {key: measurement[key] for key in MEMORY_KEYS},
            "nsys": {
                key: value for key, value in nsys.items() if key != "kernel_names"
            },
            "accuracy": accuracy,
            "within_draft": within_draft(data.model, accuracy),
        },
    )


def historical_ncu(data, condition, measurement):
    dino = data.model == DINO
    bundle = "run-i2304-dino-tensorrt" if dino else "run-i2303-vit-tensorrt"
    path = RUNS / bundle / "metrics.json"
    results = read(path)
    if dino:
        keys = (("variants" if condition == "2-plugin" else "conditions"), condition)
        identity_keys = ("model", "bucket", "checkpoint_sha256", "tensor_set_sha256")
    else:
        keys = ("models", data.model, condition)
        identity_keys = ("model", "checkpoint_sha256", "tensors")
    old = results
    for key in keys:
        old = old[key]
    identity = old["identity"]
    if (
        identity["profile_address"] != data.identity["profile_anchor"]
        or any(identity[key] != data.identity[key] for key in identity_keys)
        or old["measurement"]["batch_shape"] != [1, *data.input_shape]
        or any(
            old["measurement"][key] != measurement[key]
            for key in ("device_uuid", "driver", "torch", "cuda")
        )
    ):
        raise ValueError(
            f"Historical B=1 comparison is from a different model, input or GPU: {keys}"
        )
    value = {
        "artifact": f"../{bundle}/metrics.json",
        "case_keys": list(keys),
        "nsys_kernels": old["profiles"]["nsys"]["gpu_kernels"],
        "engine_sha256": old.get(
            "engine_sha256", old.get("conversion", {}).get("plan_sha256")
        ),
    }
    counters = old["profiles"].get("ncu")
    if counters is None and condition == "2-plugin":
        return {
            **value,
            "status": "unavailable",
            "reason": "No B=1 ncu in #2304 plugin comparison",
        }
    if (
        counters["status"] != "measured"
        or counters["profiled_kernels"] != value["nsys_kernels"]
    ):
        raise ValueError(f"Historical B=1 profiler coverage differs: {keys}")
    job = old["profile_queue_jobs"]["ncu"] if dino else old["profile_queue_job"]
    return {**value, **counters, "queue_job": f"{job}.job", "replay_mode": "kernel"}


def representative_ncu(root, model, condition, case, measurement):
    name = f"{LABELS[model]}-ncu-{condition}"
    attempts = []
    for retry in ("retry-2", "retry-3"):
        path = root / retry / name / "ncu.json"
        if not path.is_file():
            continue
        receipt = read(path)
        if (
            receipt["measurement"] != measurement
            or Path(receipt["source"]).resolve() != case
            or receipt["nsys_kernels"] != measurement["nsys"]["gpu_kernels"]
        ):
            raise ValueError(
                f"Profiler was captured from a different measurement: {path}"
            )
        saved, csv_path = receipt["ncu"], path.with_name("ncu.csv")
        proof = {
            "artifact": str(path.relative_to(root)),
            "queue_job": f"{receipt['queue_job']}.job",
            "status": saved["status"],
            "replay_mode": receipt["replay_mode"],
            "timeout_seconds": receipt["timeout_seconds"],
            "nsys_kernels": receipt["nsys_kernels"],
        }
        if saved["status"] == "measured":
            counters = read_ncu(csv_path)
            if (
                counters["profiled_kernels"] != proof["nsys_kernels"]
                or any(saved[key] != value for key, value in counters.items())
                or ncu_launch_shapes(*ncu_rows(csv_path))
                != nsys_launch_shapes(case / "nsys.sqlite")
            ):
                raise ValueError(
                    f"Profiler counters or ordered kernel coverage changed: {path}"
                )
            proof.update(
                counters,
                raw_csv_sha256=sha256_file(csv_path),
                ordered_launch_shapes_match=True,
            )
        else:
            reason = saved["reason"]
            proof["reason"] = "timeout" if "timed out after" in reason else reason
        attempts.append(proof)
    if not attempts:
        raise ValueError(f"No ncu attempt was recorded for {case}")
    selected = next(
        (attempt for attempt in reversed(attempts) if attempt["status"] == "measured"),
        attempts[-1],
    )
    return {
        **selected,
        "other_attempts": [attempt for attempt in attempts if attempt is not selected],
    }


def summarize(root, vit_inputs, dino_inputs):
    models, environment = {}, None
    for model, batches in BATCHES.items():
        data = PinnedInputs(model, dino_inputs if model == DINO else vit_inputs / model)
        baseline = root / model / "c0-b1"
        reference = data.predictions(baseline)
        reference_sha = sha256_file(baseline / data.prediction_name)
        conditions = {}
        for condition in CONDITIONS[model]:
            cases = {}
            plan_check = (
                None if condition in {"0", "1"} else check_plan(root, data, condition)
            )
            for batch in batches:
                case, result, measurement, summary = check_case(
                    root,
                    data,
                    condition,
                    batch,
                    reference,
                    reference_sha,
                    plan_check,
                )
                device = {key: measurement[key] for key in DEVICE_KEYS}
                if environment is not None and device != environment:
                    raise ValueError(
                        f"Measured GPU or software identity changed: {case}"
                    )
                environment = device
                cases[str(batch)] = summary
                if batch == 1:
                    batch1 = historical_ncu(data, condition, measurement)
                if batch == OPT_BATCH[model]:
                    ncu = representative_ncu(root, model, condition, case, result)
            conditions[condition] = {
                "batches": cases,
                "engine": plan_check[1] if plan_check is not None else None,
                "ncu_batch1": batch1,
                "ncu_representative": {"batch_size": OPT_BATCH[model], **ncu},
            }
        models[model] = {
            "identity": data.identity,
            "evaluation_items": len(data.addresses),
            "fp32_reference_prediction_sha256": reference_sha,
            "conditions": conditions,
        }
    return {"environment": environment, "models": models}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--vit-inputs", type=Path, required=True)
    parser.add_argument("--dino-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(
        args.output,
        summarize(args.source.resolve(strict=True), args.vit_inputs, args.dino_inputs),
    )


if __name__ == "__main__":
    main()
