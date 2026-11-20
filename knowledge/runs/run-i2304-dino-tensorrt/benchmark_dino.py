"""Run DINO timing, decoded accuracy, or profilers in isolated queue processes."""

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from benchmark import NSYS, run_stage, write_json
from convert import CONDITIONS, TRT_PRECISIONS, require_queue
from dino_accuracy import compare_dino_frame
from fixed_inputs import (
    CHECKPOINT,
    HERE,
    LOCK,
    PINS,
    require_bucket,
    rows_for,
    sha256_file,
    tensor_digest,
)
from profiles import capture

CHECKPOINT_SHA256 = "e61688afe3af91b25955e9f9601d04b42068327a2e076ee871828089aa4ffed5"
EXTENSION_SHA256 = "68580f2e5b0454dba630ba3d37e55e383d5babd86b15cda92d4e7c104f5797b7"
NCU_TIMEOUT_SECONDS = 45 * 60


def distribution(values):
    if not values.size:
        return None
    return {
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def compare(reference, candidate, rows):
    order = [row["file"] for row in rows]
    if any(
        [detection["file"] for detection in data] != order
        for data in (reference, candidate)
    ):
        raise ValueError("DINO detection order differs from the fixed input lock")
    groups = defaultdict(list)
    for row, ref, result in zip(rows, reference, candidate, strict=True):
        match = compare_dino_frame(
            np.asarray(ref["boxes"]).reshape(-1, 4),
            ref["scores"],
            np.asarray(result["boxes"]).reshape(-1, 4),
            result["scores"],
        )
        groups["all"].append(match)
        groups[f"{row['clip']}/{row['camera']}"].append(match)

    def summarize(matches):
        iou = np.concatenate([item["iou"] for item in matches])
        xyxy = np.concatenate([item["xyxy_abs_error_px"] for item in matches])
        missing = sum(item["missing"] for item in matches)
        extra = sum(item["extra"] for item in matches)
        return {
            "images": len(matches),
            "matched": len(iou),
            "reference_boxes": len(iou) + missing,
            "candidate_boxes": len(iou) + extra,
            "missing": missing,
            "extra": extra,
            "iou": distribution(iou),
            "xyxy_abs_error_px": distribution(xyxy),
            "xyxy_components_abs_error_px": {
                name: distribution(xyxy[:, i])
                for i, name in enumerate(("x1", "y1", "x2", "y2"))
            },
            "score_abs_error": distribution(
                np.concatenate([item["score_abs_error"] for item in matches])
            ),
        }

    return {
        "overall": summarize(groups.pop("all")),
        "by_clip_camera": {
            group: summarize(matches) for group, matches in groups.items()
        },
    }


def verify_inputs(directory, bucket, checkpoint):
    lock = json.loads(LOCK.read_text())
    require_bucket(lock, bucket)
    if json.loads((directory / "tensors.json").read_text()) != lock:
        raise ValueError("DINO tensor receipt does not match the fixed input lock")
    if sha256_file(PINS) != lock["pins_sha256"]:
        raise ValueError("Clip pin document changed")
    if tensor_digest(directory, lock["samples"]) != lock["tensor_set_sha256"]:
        raise ValueError("DINO tensor set digest changed")
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("DINO checkpoint digest changed")
    return lock


def check_extension():
    binary = Path(os.environ["TENNIS_LAB_DINO_EXTENSION"])
    if not binary.is_absolute() or sha256_file(binary) != EXTENSION_SHA256:
        raise ValueError(
            "TENNIS_LAB_DINO_EXTENSION must identify the pinned #2238 binary"
        )


def run_case(args):
    lock = verify_inputs(args.inputs, args.bucket, args.checkpoint)
    if args.condition not in TRT_PRECISIONS:
        check_extension()
    identity = {
        "model": "dino_4scale_swin_l",
        "bucket": args.bucket,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "tensor_set_sha256": lock["tensor_set_sha256"],
        "plan_profile": lock["buckets"][args.bucket]["profile"],
        "profile_address": lock["buckets"][args.bucket]["representative"],
    }
    result = {"identity": identity, "condition": args.condition}
    command = [
        sys.executable,
        "-B",
        str(HERE / "measure_dino.py"),
        "--condition",
        args.condition,
        "--bucket",
        args.bucket,
        "--checkpoint",
        str(args.checkpoint),
        "--inputs",
        str(args.inputs),
        "--output",
        str(args.output),
    ]
    if args.engine_dir:
        plan = args.engine_dir / "dino.plan"
        conversion = json.loads((args.engine_dir / "conversion.json").read_text())
        method = conversion.get("method")
        if (
            conversion["condition"] != args.condition
            or conversion["input_shape"] != identity["plan_profile"]["opt"]
            or conversion["tensor_set_sha256"] != identity["tensor_set_sha256"]
            or conversion["checkpoint_sha256"] != identity["checkpoint_sha256"]
            or conversion["plan_sha256"] != sha256_file(plan)
            or method not in {"grid", "plugin"}
            or (method == "plugin" and args.condition != "2")
        ):
            raise ValueError("DINO TensorRT plan receipt does not match the condition")
        result["conversion"] = conversion
        command += ["--plan", str(plan), "--method", method]
    else:
        result["extension_sha256"] = EXTENSION_SHA256

    args.output.mkdir(parents=True)
    if args.profile_from:
        previous = json.loads((args.profile_from / "result.json").read_text())
        if any(previous.get(key) != value for key, value in result.items()):
            raise ValueError("Profile source used a different condition, plan or input")
        if (
            args.profiles == "ncu"
            and previous["profiles"].get("nsys", {}).get("status") != "measured"
        ):
            raise ValueError(
                "ncu follow-up requires an nsys kernel count for the same condition"
            )
        result = previous
    else:
        run_stage(command, args.output / "measure.log", timeout=30 * 60)
        detections = args.output / "detections.json"
        actual = json.loads(detections.read_text())
        baseline = actual
        if args.reference:
            reference = json.loads((args.reference / "result.json").read_text())
            path = args.reference / "detections.json"
            if reference["identity"] != identity or reference["condition"] != "0":
                raise ValueError(
                    "Baseline must be condition 0 with the same input identity"
                )
            if reference["detections_sha256"] != sha256_file(path):
                raise ValueError("Baseline detection digest changed")
            baseline = json.loads(path.read_text())
        result["measurement"] = json.loads(
            (args.output / "measurement.json").read_text()
        )
        result["accuracy"] = compare(
            baseline, actual, rows_for(lock, args.bucket, "eval")
        )
        result["detections_sha256"] = sha256_file(detections)
        result["profiles"] = {}
        result["profile_queue_jobs"] = {}
    write_json(args.output / "result.json", result)
    for tool, executable in (("nsys", args.nsys), ("ncu", args.ncu)):
        if args.profiles in ("all", tool):
            result["profiles"][tool] = capture(
                tool,
                str(executable),
                [*command, "--profile"],
                args.output,
                ncu_timeout=args.ncu_timeout,
                ncu_replay=args.ncu_replay,
            )
            if tool == "ncu":
                result["profiles"][tool]["replay_mode"] = args.ncu_replay
                result["profiles"][tool]["timeout_s"] = args.ncu_timeout
            result["profile_queue_jobs"][tool] = os.environ["TENNIS_RUN_ID"]
            write_json(args.output / "result.json", result)

    nsys = result["profiles"].get("nsys", {})
    ncu = result["profiles"].get("ncu", {})
    if (
        nsys.get("status") == ncu.get("status") == "measured"
        and nsys["gpu_kernels"] != ncu["profiled_kernels"]
    ):
        ncu.update(
            status="coverage_mismatch",
            reason=f"ncu={ncu['profiled_kernels']} kernels, nsys={nsys['gpu_kernels']} kernels",
        )
        write_json(args.output / "result.json", result)
    if any(profile["status"] != "measured" for profile in result["profiles"].values()):
        return 3
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--engine-dir", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--profile-from", type=Path)
    parser.add_argument(
        "--profiles", choices=("all", "nsys", "ncu", "none"), default="all"
    )
    parser.add_argument("--nsys", type=Path, default=NSYS)
    parser.add_argument("--ncu", type=Path, default=Path("/usr/local/cuda/bin/ncu"))
    parser.add_argument("--ncu-timeout", type=int, default=NCU_TIMEOUT_SECONDS)
    parser.add_argument(
        "--ncu-replay", choices=("kernel", "application"), default="kernel"
    )
    args = parser.parse_args()
    if args.ncu_timeout <= 0:
        parser.error("--ncu-timeout must be positive")
    if args.ncu_replay == "application" and not (
        args.profile_from and args.profiles == "ncu"
    ):
        parser.error(
            "Application replay is a diagnostic follow-up to a pinned measurement"
        )
    if (args.condition in TRT_PRECISIONS) != (args.engine_dir is not None):
        parser.error("Specify --engine-dir only for TensorRT conditions")
    if (args.condition != "0" and not args.profile_from) != (
        args.reference is not None
    ):
        parser.error(
            "This comparison requires --reference only for new non-baseline measurements"
        )
    require_queue()
    try:
        return run_case(args)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError):
        if args.output.exists() and not (args.output / "result.json").exists():
            write_json(args.output / "status.json", {"status": "failed"})
        raise


if __name__ == "__main__":
    sys.exit(main())
