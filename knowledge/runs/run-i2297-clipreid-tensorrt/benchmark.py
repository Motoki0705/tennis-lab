"""Run one pinned model/precision condition, compare it with FP32, and capture profiles."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from convert import CONDITIONS, PLAN, require_queue
from inputs import (
    CHECKPOINT,
    CHECKPOINT_SHA256,
    HERE,
    MODEL,
    check_split,
    eval_addresses,
    sha256_file,
)
from profiles import capture

PROFILE_ADDRESS = ("video_000/clip_000", "cam0", 178, 0)
NSYS = Path(sys.prefix) / "nsight-dist/opt/nvidia/nsight-systems/2026.5.2/bin/nsys"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_stage(command, log, timeout):
    with log.open("x") as stream:
        print(subprocess.list2cmdline(command), file=stream, flush=True)
        subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout,
        )


def normalize(features):
    norm = np.linalg.norm(features.astype(np.float64), axis=-1, keepdims=True)
    if not np.isfinite(norm).all() or np.any(norm <= 0):
        raise ValueError("Nonfinite or zero feature norms")
    return features / norm


def nearest_ids(embedding, addresses):
    gallery = {}
    for index, (clip, camera, _frame, player) in enumerate(addresses):
        gallery.setdefault((clip, camera, player), index)
    labels = []
    for index, (clip, camera, _frame, _player) in enumerate(addresses):
        candidates = [
            (key[2], row)
            for key, row in gallery.items()
            if key[0] == clip and key[1] != camera
        ]
        if len({player for player, _ in candidates}) < 2:
            raise ValueError(f"Insufficient cross-camera gallery IDs in {clip}")
        rows = [row for _, row in candidates]
        chosen = int(np.argmax(embedding[rows] @ embedding[index]))
        labels.append(candidates[chosen][0])
    return np.asarray(labels)


def accuracy(actual, baseline, addresses):
    if (
        actual.shape != baseline.shape
        or actual.ndim != 2
        or len(actual) != len(addresses)
    ):
        raise ValueError("Feature outputs are not aligned to the fixed observations")
    trial, reference = normalize(actual), normalize(baseline)
    cosine = np.clip(np.einsum("ij,ij->i", trial, reference), -1, 1)
    pred, gold = nearest_ids(trial, addresses), nearest_ids(reference, addresses)
    changed = [
        {"address": addr, "fp32_id": int(before), "condition_id": int(after)}
        for addr, before, after in zip(addresses, gold, pred, strict=True)
        if before != after
    ]
    return {
        "crops": len(cosine),
        "cosine_mean": float(cosine.mean()),
        "cosine_median": float(np.median(cosine)),
        "cosine_p05": float(np.quantile(cosine, 0.05)),
        "cosine_min": float(cosine.min()),
        "id_matches_fp32": len(cosine) - len(changed),
        "id_changed_examples": changed,
        "id_matches_association": int(
            np.sum(pred == np.asarray([row[3] for row in addresses]))
        ),
    }


def verify_inputs(directory):
    lock = json.loads((HERE / "tensor-lock.json").read_text())
    if json.loads((directory / "tensors.json").read_text()) != lock:
        raise ValueError("Prepared input receipt differs from tensor-lock.json")
    pins = HERE / "inputs.json"
    if sha256_file(pins) != lock["pins_sha256"]:
        raise ValueError("Crop address manifest differs from tensor-lock.json")
    for split, info in lock["tensors"].items():
        if sha256_file(directory / f"{split}.npy") != info["sha256"]:
            raise ValueError(f"{split} tensor SHA-256 mismatch")
    book = json.loads(pins.read_text())
    check_split(book)
    return lock, eval_addresses(book)


def run_case(args):
    lock, addresses = verify_inputs(args.inputs)
    if sha256_file(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Unexpected CLIP-ReID checkpoint contents")
    profile_index = addresses.index(PROFILE_ADDRESS)
    identity = {
        "model": args.model,
        "condition": args.condition,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "tensors": lock["tensors"],
        "profile_address": list(PROFILE_ADDRESS),
    }
    comparable_keys = ("model", "checkpoint_sha256", "tensors", "profile_address")

    def verify_reference(path, expected_condition):
        receipt = json.loads((path / "result.json").read_text())
        before = receipt["identity"]
        if before["condition"] != expected_condition or any(
            before[key] != identity[key] for key in comparable_keys
        ):
            raise ValueError(f"Reference identity mismatch: {path}")
        if sha256_file(path / "features.npy") != receipt["features_sha256"]:
            raise ValueError(f"Reference features do not match their receipt: {path}")
        return receipt

    output = args.output
    if args.profile_from:
        measured = args.profile_from
        copied = verify_reference(measured, args.condition)
        if (
            args.condition in {"2", "3a", "3b"}
            and sha256_file(measured / PLAN) != copied["engine_sha256"]
        ):
            raise ValueError("Existing TensorRT engine does not match its receipt")
        result = {**copied, "profile_from": str(measured)}
    else:
        measured = output
        result = {"identity": identity}
        if args.condition != "0":
            verify_reference(args.reference, "0")
        if args.condition in {"2", "3a", "3b"}:
            run_stage(
                [
                    sys.executable,
                    "-B",
                    str(HERE / "convert.py"),
                    "--condition",
                    args.condition,
                    "--checkpoint",
                    str(args.checkpoint),
                    "--calibration",
                    str(args.inputs / "calibration.npy"),
                    "--output",
                    str(output),
                ],
                output / "convert.log",
                timeout=50 * 60,
            )
            result["engine_sha256"] = sha256_file(output / PLAN)
            result["conversion"] = json.loads((output / "conversion.json").read_text())

    command = [
        sys.executable,
        "-B",
        str(HERE / "runtime.py"),
        "--condition",
        args.condition,
        "--checkpoint",
        str(args.checkpoint),
        "--eval",
        str(args.inputs / "eval.npy"),
        "--output",
        str(measured),
    ]
    if not args.profile_from:
        run_stage(command, output / "measure.log", timeout=180)
        result["measurement"] = json.loads((output / "measurement.json").read_text())
        actual = np.load(output / "features.npy", allow_pickle=False)
        baseline = (
            actual
            if args.condition == "0"
            else np.load(args.reference / "features.npy", allow_pickle=False)
        )
        result["accuracy"] = accuracy(actual, baseline, addresses)
        result["features_sha256"] = sha256_file(output / "features.npy")
        result["measurement_queue_job"] = os.environ["TENNIS_RUN_ID"]

    result["profile_queue_job"] = os.environ["TENNIS_RUN_ID"]
    result["profiles"] = {}
    write_json(output / "result.json", result)
    for tool, executable in {"nsys": str(args.nsys), "ncu": str(args.ncu)}.items():
        if args.profiles in {"all", tool}:
            result["profiles"][tool] = capture(
                tool,
                executable,
                [*command, "--profile-index", str(profile_index)],
                output,
            )
            write_json(output / "result.json", result)
    nsys = result["profiles"].get("nsys", {})
    ncu = result["profiles"].get("ncu", {})
    if nsys.get("status") == ncu.get("status") == "measured" and (
        nsys["gpu_kernels"] != ncu["profiled_kernels"]
    ):
        ncu["status"] = "coverage_mismatch"
        ncu["reason"] = "The two profiles cover different numbers of kernels."
        write_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=(MODEL,))
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument(
        "--reference", type=Path, help="condition 0 output for accuracy comparison"
    )
    parser.add_argument(
        "--profiles", choices=("all", "none", "nsys", "ncu"), default="all"
    )
    parser.add_argument("--nsys", type=Path, default=NSYS)
    parser.add_argument("--ncu", type=Path, default=Path("/usr/local/cuda/bin/ncu"))
    parser.add_argument(
        "--profile-from",
        type=Path,
        help="reuse measurements and engine from an existing result",
    )
    args = parser.parse_args()
    if args.reference is None and args.condition != "0" and not args.profile_from:
        parser.error("--reference is required for a comparison with FP32")
    if args.profile_from and args.profiles == "none":
        parser.error("--profile-from requires selecting a profiler")
    require_queue()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = run_case(args)
    except Exception as exc:
        write_json(args.output / "status.json", {"status": "failed", "error": str(exc)})
        raise
    partial = any(tool["status"] != "measured" for tool in result["profiles"].values())
    status = "profile_incomplete" if partial else "complete"
    write_json(args.output / "status.json", {"status": status})
    print(
        f"{args.model} condition {args.condition}: {status} ({args.output})", flush=True
    )
    return 3 if partial else 0


if __name__ == "__main__":
    sys.exit(main())
