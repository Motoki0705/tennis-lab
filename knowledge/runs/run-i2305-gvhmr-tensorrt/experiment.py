"""Use the shared TensorRT protocol for one GVHMR condition and shape."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from benchmark import NSYS, run_stage, write_json
from convert import CONDITIONS, TRT_PRECISIONS, require_queue
from inputs import sha256_file
from locked_inputs import HERE
from profiles import capture
from run_gvhmr import read_conversion


def runtime_command(args, *, mode):
    command = [
        sys.executable,
        "-B",
        str(HERE / "run_gvhmr.py"),
        "--condition",
        args.condition,
        "--length",
        str(args.length),
        "--inputs",
        str(args.inputs),
        "--checkpoints",
        str(args.checkpoints),
        "--mode",
        mode,
    ]
    if args.conversion is not None:
        command.extend(("--conversion", str(args.conversion)))
    return command


def measure(args):
    run_stage(
        [*runtime_command(args, mode="measure"), "--output", str(args.output)],
        args.output / "measure.log",
        900,
    )
    command = [
        sys.executable,
        "-B",
        str(HERE / "accuracy_gvhmr.py"),
        "--condition",
        args.condition,
        "--length",
        str(args.length),
        "--inputs",
        str(args.inputs),
        "--checkpoints",
        str(args.checkpoints),
        "--output",
        str(args.output),
    ]
    if args.reference:
        command.extend(("--reference", str(args.reference)))
    run_stage(command, args.output / "accuracy.log", 300)


def collect_profiles(args):
    measurement_root = args.profile_from or args.output
    timing = json.loads((measurement_root / "measurement.json").read_text())
    if (
        timing["condition"] != args.condition
        or timing["length"] != args.length
        or timing["tensor_lock_sha256"] != sha256_file(HERE / "tensor-lock.json")
    ):
        raise ValueError("Profiling must use the measured condition and input identity")
    if args.conversion is not None:
        expected = read_conversion(args.conversion, args.condition)["engine_sha256"]
        if timing["engine_sha256"] != expected:
            raise ValueError("Profiling must use the same TensorRT plan as timing")
    profiling = {}
    command = runtime_command(args, mode="profile")
    for tool, executable in (("nsys", str(NSYS)), ("ncu", shutil.which("ncu"))):
        profiling[tool] = (
            capture(tool, executable, command, args.output)
            if executable is not None
            else {"status": "unavailable", "reason": f"{tool} not found"}
        )
    nsys, ncu = profiling["nsys"], profiling["ncu"]
    if (
        nsys["status"] == "measured"
        and ncu["status"] == "measured"
        and nsys["gpu_kernels"] != ncu["profiled_kernels"]
    ):
        ncu["status"] = "coverage_mismatch"
        ncu["nsys_gpu_kernels"] = nsys["gpu_kernels"]
    profiling["queue_run_id"] = os.environ["TENNIS_RUN_ID"]
    profiling["measurement_sha256"] = sha256_file(measurement_root / "measurement.json")
    write_json(args.output / "profiles.json", profiling)
    return all(profiling[tool]["status"] == "measured" for tool in ("nsys", "ncu"))


def probe_boundaries(args):
    profile = json.loads((HERE / "shape-profile.json").read_text())
    evidence = {}
    in_range = profile["profile_lengths"].values()
    for frames in (*in_range, *profile["outside_lengths"]):
        args.length = frames
        command = runtime_command(args, mode="probe")
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=120,
        )
        log = args.output / f"L{frames}.log"
        log.write_text(completed.stdout)
        rejected_by_trt = (
            f"TensorRT rejected input shape obs=(1, {frames}, 17, 3)"
            in completed.stdout
        )
        valid_evidence = (
            completed.returncode == 0
            and f"Engine executed L={frames}" in completed.stdout
            if frames in in_range
            else completed.returncode != 0 and rejected_by_trt
        )
        evidence[str(frames)] = {
            "exit_code": completed.returncode,
            "in_profile": frames in in_range,
            "rejected_by_trt": rejected_by_trt,
            "evidence_valid": valid_evidence,
            "log": log.name,
        }
        write_json(args.output / "boundaries.json", evidence)
        if not valid_evidence:
            raise RuntimeError(
                f"Unexpected TensorRT boundary result for L={frames}: {log}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--length", type=int)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--checkpoints", default=Path("ckpt"), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--conversion", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--profiles", choices=("both", "none"), default="both")
    parser.add_argument("--profile-from", type=Path)
    parser.add_argument("--boundaries", action="store_true")
    args = parser.parse_args()
    if (args.condition in TRT_PRECISIONS) != (args.conversion is not None):
        parser.error("TensorRT conditions require a conversion directory")
    if args.boundaries:
        if args.conversion is None or args.length is not None:
            parser.error(
                "Boundary probes require a TensorRT plan and use the pinned lengths"
            )
    elif args.length is None:
        parser.error("Measurement requires a sequence length")
    if not (
        args.boundaries or args.profile_from or args.condition == "0" or args.reference
    ):
        parser.error("Accuracy requires a condition 0 reference at the same length")
    if args.profile_from and args.profiles == "none":
        parser.error("Profiling follow-up requires profiling tools")
    require_queue()
    args.output.mkdir(parents=True)
    status = {"condition": args.condition, "length": args.length, "stage": "boundaries"}
    try:
        if args.boundaries:
            probe_boundaries(args)
        else:
            if args.profile_from is None:
                status["stage"] = "measurement"
                measure(args)
            if args.profiles == "both":
                status["stage"] = "profiling"
                if not collect_profiles(args):
                    write_json(
                        args.output / "status.json",
                        {**status, "status": "profile_incomplete"},
                    )
                    raise SystemExit(3)
    except Exception as exc:
        write_json(
            args.output / "status.json",
            {**status, "status": "failed", "error": str(exc)},
        )
        raise
    write_json(args.output / "status.json", {**status, "status": "done"})


if __name__ == "__main__":
    main()
