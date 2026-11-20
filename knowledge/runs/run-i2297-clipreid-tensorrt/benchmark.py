"""Run one pinned model/precision condition, compare it with FP32, and capture profiles."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from convert import CONDITIONS, PLAN, TRT_PRECISIONS, require_queue
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


def load_adapter(path, model):
    sys.path.insert(0, str(path.resolve().parent))
    spec = importlib.util.spec_from_file_location("experiment_adapter", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load experiment adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.adapter(model)


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
    adapter = load_adapter(args.adapter, args.model) if args.adapter else None
    extra = {}
    if adapter is None:
        lock, addresses = verify_inputs(args.inputs)
        checkpoint_sha256 = CHECKPOINT_SHA256
        profile_address = PROFILE_ADDRESS
        compare = accuracy
    else:
        lock, addresses = adapter.verify_inputs(args.inputs)
        checkpoint_sha256 = adapter.checkpoint_sha256
        profile_address = tuple(lock["profile_address"])
        extra = adapter.identity()
        compare = adapter.accuracy
    if sha256_file(args.checkpoint) != checkpoint_sha256:
        raise ValueError(f"Unexpected {args.model} checkpoint contents")
    profile_index = addresses.index(profile_address)
    identity = {
        "model": args.model,
        "condition": args.condition,
        "checkpoint_sha256": checkpoint_sha256,
        "tensors": lock["tensors"],
        "profile_address": list(profile_address),
        **extra,
    }
    comparable_keys = identity.keys() - {"condition"}
    adapter_args = (
        ["--adapter", str(args.adapter.resolve()), "--model", args.model]
        if adapter
        else []
    )

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
    source = args.profile_from or args.measure_from
    previous = verify_reference(source, args.condition) if source else None
    nsys_reference = {}
    if args.profile_from:
        result = {**previous, "profile_from": str(source)}
        if args.profiles == "ncu":
            nsys_reference = previous["profiles"].get("nsys", {})
            if nsys_reference.get("status") != "measured":
                raise ValueError(
                    "ncu follow-up requires a measured nsys result in --profile-from"
                )
            result["nsys_comparison"] = {"gpu_kernels": nsys_reference["gpu_kernels"]}
    else:
        result = {"identity": identity}
        if args.condition != "0":
            verify_reference(args.reference, "0")

    if args.condition in TRT_PRECISIONS:
        if previous is not None:
            plan = source / previous.get("engine_path", PLAN)
            if sha256_file(plan) != previous["engine_sha256"]:
                raise ValueError("Existing TensorRT engine does not match its receipt")
            result["engine_sha256"] = previous["engine_sha256"]
            result["conversion"] = previous["conversion"]
            if args.measure_from:
                result["measurement_from"] = str(source)
        else:
            plan = output / (f"{args.model}.plan" if adapter else PLAN)
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
                    *adapter_args,
                    *(["--ptq-recipe", args.ptq_recipe] if args.ptq_recipe else []),
                ],
                output / "convert.log",
                timeout=50 * 60,
            )
            result["engine_sha256"] = sha256_file(plan)
            result["conversion"] = json.loads((output / "conversion.json").read_text())
        if adapter is not None:
            digest = sha256_file(plan.with_suffix(".precision.json"))
            if previous is not None and previous["precision_manifest_sha256"] != digest:
                raise ValueError(
                    "Existing engine precision manifest differs from its receipt"
                )
            result["precision_manifest_sha256"] = digest
        result["engine_path"] = os.path.relpath(plan, output)

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
        str(output),
    ]
    if args.condition in TRT_PRECISIONS:
        command += ["--plan", str(plan)]
    command.extend(adapter_args)
    if not args.profile_from:
        run_stage(command, output / "measure.log", timeout=360)
        result["measurement"] = json.loads((output / "measurement.json").read_text())
        actual = np.load(output / "features.npy", allow_pickle=False)
        baseline = (
            actual
            if args.condition == "0"
            else np.load(args.reference / "features.npy", allow_pickle=False)
        )
        result["accuracy"] = compare(actual, baseline, addresses)
        result["features_sha256"] = sha256_file(output / "features.npy")
        result["measurement_queue_job"] = os.environ["TENNIS_RUN_ID"]

    result["profile_queue_job"] = os.environ["TENNIS_RUN_ID"]
    result["profiles"] = {}
    write_json(output / "result.json", result)
    for tool, executable in {"nsys": str(args.nsys), "ncu": str(args.ncu)}.items():
        if args.profiles in {"all", tool}:
            profile_command = [*command, "--profile-index", str(profile_index)]
            if tool == "ncu" and args.condition in TRT_PRECISIONS:
                # ncu 2025.3.1 raw CSV misquotes TensorRT's JSON NVTX payloads.
                profile_command.append("--omit-trt-layer-nvtx")
            result["profiles"][tool] = capture(
                tool, executable, profile_command, output, ncu_timeout=args.ncu_timeout
            )
            if tool == "ncu":
                result["profiles"][tool]["timeout_seconds"] = args.ncu_timeout
            write_json(output / "result.json", result)
    nsys = result["profiles"].get("nsys", nsys_reference)
    ncu = result["profiles"].get("ncu", {})
    if nsys.get("status") == ncu.get("status") == "measured" and (
        nsys["gpu_kernels"] != ncu["profiled_kernels"]
    ):
        ncu["status"] = "coverage_mismatch"
        ncu["reason"] = (
            f"Nsight Compute covered {ncu['profiled_kernels']} kernels, "
            f"Nsight Systems covered {nsys['gpu_kernels']}."
        )
        write_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--adapter", type=Path, help="model adapter for a follow-up experiment"
    )
    parser.add_argument(
        "--ptq-recipe", help="explicit PTQ recipe provided by the adapter"
    )
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
    parser.add_argument("--ncu-timeout", type=int, default=900)
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument(
        "--profile-from",
        type=Path,
        help="reuse measurements and engine from an existing result",
    )
    reuse.add_argument(
        "--measure-from",
        type=Path,
        help="repeat timing with the existing TensorRT engine from a result",
    )
    args = parser.parse_args()
    if args.ncu_timeout <= 0:
        parser.error("--ncu-timeout must be positive")
    if (args.model != MODEL or args.ptq_recipe) and args.adapter is None:
        parser.error("--adapter is required for this model or PTQ recipe")
    if args.ptq_recipe and (
        args.condition not in {"3a", "3b"} or args.profile_from or args.measure_from
    ):
        parser.error("--ptq-recipe requires a new quantized engine build")
    if args.reference is None and args.condition != "0" and not args.profile_from:
        parser.error("--reference is required for a comparison with FP32")
    if args.profile_from and args.profiles == "none":
        parser.error("--profile-from requires selecting a profiler")
    if args.measure_from and args.condition not in TRT_PRECISIONS:
        parser.error("--measure-from requires a TensorRT condition")
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
