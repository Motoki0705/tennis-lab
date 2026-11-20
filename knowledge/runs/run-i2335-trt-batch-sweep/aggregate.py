"""Promote verified timing, accuracy and profiler receipts from the sweep outputs."""

import argparse
import json
from pathlib import Path

import numpy as np
from benchmark import accuracy, verify_inputs, write_json
from clip_ptq import accuracy_excluding_selection
from inputs import sha256_file
from profiles import read_ncu
from sweep import (
    CLIP,
    CLIP_BATCHES,
    CLIP_OPT,
    GVHMR,
    GVHMR_BATCHES,
    GVHMR_LENGTHS,
    GVHMR_OPT,
    VIT,
    case_id,
)


def read(path):
    return json.loads(path.read_text())


def measurement_summary(measurement):
    summary = {
        key: value
        for key, value in measurement.items()
        if key not in {"latency_samples_ms", "nvml_device_snapshots_bytes"}
    }
    if "tail_batch" in summary:
        summary["tail_batch"] = measurement_summary(summary["tail_batch"])
    return summary


def unavailable(path, condition, *, fallback):
    root = path.parent if (path.parent / "sweep.json").is_file() else fallback
    report = read(root / "sweep.json")
    failure = next(
        (
            dict(stage, name=name)
            for name, stage in report["stages"].items()
            if not stage["verified"]
            and (
                name in {path.name, f"build-{condition}"}
                or name.startswith(f"probe-c{condition}-")
            )
        ),
        None,
    )
    return {
        "status": "unavailable",
        "stage_evidence": str(root / "sweep.json"),
        "failure": failure,
    }


def ncu_summary(path, receipt, expected, job):
    stored = receipt["ncu"]
    row = {
        "status": stored["status"],
        "artifact": str(path),
        "queue_job": job,
        "nsys_kernels": expected,
    }
    if stored["status"] != "measured":
        row["timeout_seconds"] = stored.get("timeout_seconds")
        reason = stored.get("reason")
        row["reason"] = "timeout" if reason and "timed out after" in reason else reason
        return row
    csv_path = path.parent / "ncu.csv"
    counters = read_ncu(csv_path)
    if counters["profiled_kernels"] != expected or any(
        stored[key] != value for key, value in counters.items()
    ):
        raise ValueError(f"ncu counters or kernel coverage changed: {path}")
    return {**stored, **row, "raw_csv_sha256": sha256_file(csv_path)}


def clip_cases(root, addresses, tensors, prior_vit):
    retry2, retry3 = root / "retry-2", root / "retry-3"
    reference = retry2 / "clipreid/c0-b1/features.npy"
    fp32_ref = sha256_file(reference)
    baseline = np.load(reference, allow_pickle=False)
    selected = {
        tuple(row)
        for row in prior_vit["clip_accuracy_excluding_selection"]["excluded_addresses"]
    }
    if len(selected) != 12 or not selected <= set(addresses):
        raise ValueError(
            "CLIP PTQ selection must name exactly 12 of the evaluation crops"
        )
    cases = {}
    for condition in ("0", "1", "2", "3a", "3b"):
        latest = retry3 if condition == "3b" else retry2
        for batch in CLIP_BATCHES:
            name = case_id(condition, batch, None)
            case = latest / "clipreid" / name
            path = case / "result.json"
            if not path.is_file():
                cases[name] = unavailable(case, condition, fallback=retry2 / "clipreid")
                continue
            stored = read(path)
            identity = stored["identity"]
            prediction = case / "features.npy"
            if (
                (identity["condition"], identity["batch_size"], identity["tensors"])
                != (condition, batch, tensors)
                or stored["features_sha256"] != sha256_file(prediction)
                or stored.get(
                    "fp32_reference_features_sha256", stored["features_sha256"]
                )
                != fp32_ref
            ):
                raise ValueError(f"CLIP feature or reference identity mismatch: {path}")
            actual = np.load(prediction, allow_pickle=False)
            accuracy_all = accuracy(actual, baseline, addresses)
            if json.dumps(accuracy_all, sort_keys=True) != json.dumps(
                stored["accuracy"], sort_keys=True
            ):
                raise ValueError(f"CLIP accuracy receipt mismatch: {path}")
            held_out = (
                accuracy_excluding_selection(actual, baseline, addresses, selected)
                if condition in {"3a", "3b"}
                else None
            )
            judgement = held_out or accuracy_all
            nsys = stored["profiles"]["nsys"]
            row = {
                "status": "measured" if nsys["status"] == "measured" else "partial",
                "artifact": str(path),
                "measurement_queue_job": stored["measurement_queue_job"],
                "measurement": measurement_summary(stored["measurement"]),
                "accuracy_all_crops": accuracy_all,
                "accuracy_excluding_ptq_selection": held_out,
                "within_draft": judgement["cosine_p05"] >= 0.995
                and judgement["id_matches_fp32"] / judgement["crops"] >= 0.99,
                "nsys": {
                    key: value for key, value in nsys.items() if key != "kernel_names"
                },
                "engine_sha256": stored.get("engine_sha256"),
            }
            if batch == CLIP_OPT:
                profile_path = retry3 / "ncu/clipreid" / name / "result.json"
                if not profile_path.is_file():
                    profile_path = retry2 / "ncu/clipreid" / name / "result.json"
                if profile_path.is_file():
                    profile = read(profile_path)
                    if any(
                        profile.get(key) != stored.get(key)
                        for key in ("identity", "features_sha256", "engine_sha256")
                    ):
                        raise ValueError(
                            f"CLIP ncu input or plan mismatch: {profile_path}"
                        )
                    row["ncu"] = ncu_summary(
                        profile_path,
                        profile["profiles"],
                        nsys["gpu_kernels"],
                        profile["profile_queue_job"],
                    )
                    if "ncu_prefix" in profile:
                        row["ncu"]["prefix_evidence"] = profile["ncu_prefix"] | {
                            "source": str(retry2 / "ncu/clipreid" / name)
                        }
            cases[name] = row
    return cases


def gvhmr_cases(root):
    retry2, retry3 = root / "retry-2", root / "retry-3"
    cases = {}
    lock = sha256_file(GVHMR / "tensor-lock.json")
    for length in GVHMR_LENGTHS:
        baseline = retry2 / "gvhmr" / case_id("0", 1, length)
        fp32_ref = sha256_file(baseline / "decoded-outputs.npz")
        for condition in ("0", "1", "2"):
            for batch in GVHMR_BATCHES:
                name = case_id(condition, batch, length)
                directory = (
                    retry3 / f"gvhmr-l{length}"
                    if condition == "2"
                    else retry2 / "gvhmr"
                )
                case = directory / name
                path = case / "measurement.json"
                accuracy_path = case / "accuracy.json"
                profile_path = case / "profiles.json"
                if not all(
                    receipt.is_file() for receipt in (path, accuracy_path, profile_path)
                ):
                    cases[name] = unavailable(
                        case, condition, fallback=retry2 / "gvhmr"
                    )
                    continue
                measured, accuracy_result = read(path), read(accuracy_path)
                identity_keys = (
                    "condition",
                    "batch_size",
                    "length",
                    "tensor_lock_sha256",
                )
                if (
                    tuple(measured[key] for key in identity_keys)
                    != (condition, batch, length, lock)
                    or measured["raw_outputs_sha256"]
                    != sha256_file(case / "raw-outputs.npz")
                    or accuracy_result["decoded_outputs_sha256"]
                    != sha256_file(case / "decoded-outputs.npz")
                    or accuracy_result["fp32_reference_sha256"] != fp32_ref
                ):
                    raise ValueError(
                        f"GVHMR output or reference identity mismatch: {path}"
                    )
                profile = read(profile_path)
                if profile["measurement_sha256"] != sha256_file(path):
                    raise ValueError(f"GVHMR nsys measurement mismatch: {profile_path}")
                nsys = profile["nsys"]
                row = {
                    "status": "measured" if nsys["status"] == "measured" else "partial",
                    "artifact": str(path),
                    "measurement": measurement_summary(measured),
                    "accuracy": {
                        key: value
                        for key, value in accuracy_result.items()
                        if key != "segments"
                    },
                    "nsys": {
                        key: value
                        for key, value in nsys.items()
                        if key != "kernel_names"
                    },
                }
                if batch == GVHMR_OPT:
                    if condition == "1":
                        old_case = root / "gvhmr" / name
                        replay = old_case / "profiles.json"
                        old = read(old_case / "measurement.json")
                        fields = (
                            *identity_keys,
                            "profile_batch_segments",
                            "device_uuid",
                            "driver",
                            "torch",
                            "cuda",
                        )
                        if (
                            any(old[key] != measured[key] for key in fields)
                            or read(replay)["nsys"]["kernel_names"]
                            != nsys["kernel_names"]
                        ):
                            raise ValueError(
                                f"Reused GVHMR ncu range differs: {replay}"
                            )
                        measurement_hash = sha256_file(old_case / "measurement.json")
                    else:
                        retry = retry3 if condition == "2" else retry2
                        replay = retry / "ncu/gvhmr" / name / "profiles.json"
                        measurement_hash = sha256_file(path)
                    if replay.is_file():
                        ncu = read(replay)
                        if ncu["measurement_sha256"] != measurement_hash:
                            raise ValueError(
                                f"GVHMR ncu measurement mismatch: {replay}"
                            )
                        row["ncu"] = ncu_summary(
                            replay, ncu, nsys["gpu_kernels"], ncu["queue_run_id"]
                        )
                cases[name] = row
    return cases


def collect(root):
    lock, addresses = verify_inputs(root / "inputs/clipreid")
    prior_clip = read(CLIP / "metrics.json")["conditions"]
    prior_gvhmr = read(GVHMR / "metrics.json")["conditions"]
    prior_vit = read(VIT.parent / "metrics.json")
    return {
        "schema_version": 1,
        "clipreid": clip_cases(root, addresses, lock["tensors"], prior_vit),
        "gvhmr": gvhmr_cases(root),
        "historical_batch1_ncu": {
            "clipreid": {
                condition: (
                    prior_clip[condition]
                    if condition in {"0", "1", "2"}
                    else prior_vit["models"]["clip"][condition]
                )["profiles"]["ncu"]
                for condition in ("0", "1", "2", "3a", "3b")
            },
            "gvhmr": {
                condition: {
                    length: entry["profiles"]["ncu"]
                    for length, entry in prior_gvhmr[condition].items()
                }
                for condition in ("0", "1", "2")
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, collect(args.source))


if __name__ == "__main__":
    main()
