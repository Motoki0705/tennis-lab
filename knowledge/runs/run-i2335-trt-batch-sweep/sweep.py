"""Run the pinned CLIP-ReID or GVHMR batch sweep inside an exclusive GPU job."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from benchmark import write_json
from convert import PLAN as CLIP_PLAN
from convert import require_queue
from inputs import CHECKPOINT as CLIP_CHECKPOINT
from inputs import MODEL as CLIP_MODEL

RUNS = Path(__file__).resolve().parents[1]
CLIP = RUNS / "run-i2297-clipreid-tensorrt"
GVHMR = RUNS / "run-i2305-gvhmr-tensorrt"
CLIP_BATCHES = (1, 4, 8, 16, 32)
CLIP_OPT = 16
GVHMR_BATCHES = (1, 2, 4)
GVHMR_OPT = 4
GVHMR_LENGTHS = (145, 177)


def python(script, *arguments):
    return [sys.executable, "-B", str(script), *map(str, arguments)]


def case_id(condition, batch, length):
    name = f"c{condition}-b{batch}"
    return name if length is None else f"{name}-l{length}"


def build_command(model, inputs, conversion):
    if model == "clipreid":
        return python(
            CLIP / "convert.py",
            "--condition",
            "2",
            "--checkpoint",
            CLIP_CHECKPOINT,
            "--output",
            conversion,
            "--batch-profile",
            CLIP_OPT,
            max(CLIP_BATCHES),
        )
    return python(
        GVHMR / "convert_gvhmr.py",
        "--condition",
        "2",
        "--inputs",
        inputs,
        "--output",
        conversion,
        "--batch-profile",
        GVHMR_OPT,
        max(GVHMR_BATCHES),
    )


def probe_command(model, batch, length, inputs, conversion):
    if model == "clipreid":
        return python(
            CLIP / "runtime.py",
            "--condition",
            "2",
            "--checkpoint",
            CLIP_CHECKPOINT,
            "--eval",
            inputs / "eval.npy",
            "--output",
            conversion,
            "--plan",
            conversion / CLIP_PLAN,
            "--dynamic-batch",
            "--probe",
            "--batch-size",
            batch,
        )
    return python(
        GVHMR / "run_gvhmr.py",
        "--condition",
        "2",
        "--inputs",
        inputs,
        "--conversion",
        conversion,
        "--mode",
        "probe",
        "--batch-size",
        batch,
        "--length",
        length,
    )


def case_command(model, condition, batch, length, inputs, root, conversion):
    case = root / case_id(condition, batch, length)
    arguments = [
        "--condition",
        condition,
        "--batch-size",
        batch,
        "--inputs",
        inputs,
        "--output",
        case,
    ]
    if (condition, batch) != ("0", 1):
        arguments.extend(("--reference", root / case_id("0", 1, length)))
    if condition == "2":
        arguments.extend(("--conversion", conversion))
    if model == "clipreid":
        return python(
            CLIP / "benchmark.py",
            *arguments,
            "--model",
            CLIP_MODEL,
            "--profiles",
            "all" if batch == CLIP_OPT else "nsys",
        )
    return python(
        GVHMR / "experiment.py",
        *arguments,
        "--length",
        length,
        "--ncu-timeout",
        2400,
        "--profiles",
        "both" if batch == GVHMR_OPT else "nsys",
    )


def run_sweep(model, inputs, root):
    (root / "logs").mkdir(parents=True)
    (Path(os.environ["TENNIS_REPRO_DIR"]) / "output_dir.txt").write_text(f"{root}\n")
    report = {"model": model, "queue_job": os.environ["TENNIS_RUN_ID"], "stages": {}}

    def run_stage(name, command, *, rejected_shape=None):
        log = root / "logs" / f"{name}.log"
        start = time.monotonic()
        with log.open("x") as stream:
            print(subprocess.list2cmdline(command), file=stream, flush=True)
            completed = subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, check=False
            )
        rejection = (
            f"TensorRT rejected input shape {rejected_shape}"
            if rejected_shape
            else None
        )
        passed = (
            completed.returncode > 0 and rejection in log.read_text(errors="replace")
            if rejection
            else completed.returncode == 0
        )
        report["stages"][name] = {
            "exit_code": completed.returncode,
            "verified": passed,
            "expected_rejection": rejection,
            "wall_seconds": time.monotonic() - start,
            "log": str(log.relative_to(root)),
        }
        write_json(root / "sweep.json", report)
        print(
            f"{model} {name}: exit {completed.returncode}, verified={passed}",
            flush=True,
        )
        return passed

    conversion = root / "engine"
    batches, lengths = (
        (CLIP_BATCHES, (None,))
        if model == "clipreid"
        else (GVHMR_BATCHES, GVHMR_LENGTHS)
    )
    for condition in ("0", "1", "2"):
        if condition == "2":
            if not run_stage("build", build_command(model, inputs, conversion)):
                break
            if model == "clipreid":
                probes = (
                    (1, None, None),
                    (CLIP_OPT, None, None),
                    (max(batches), None, None),
                    (0, None, "pixels=(0, 3, 256, 128)"),
                    (
                        max(batches) + 1,
                        None,
                        f"pixels=({max(batches) + 1}, 3, 256, 128)",
                    ),
                )
            else:
                short, long = GVHMR_LENGTHS
                probes = (
                    (1, short, None),
                    (GVHMR_OPT, short, None),
                    (max(batches), long, None),
                    (0, short, "length=(0,)"),
                    (max(batches) + 1, short, f"length=({max(batches) + 1},)"),
                    (1, short - 1, f"obs=(1, {short - 1}, 17, 3)"),
                    (1, long + 1, f"obs=(1, {long + 1}, 17, 3)"),
                )
            for batch, length, rejection in probes:
                passed = run_stage(
                    f"probe-{case_id(condition, batch, length)}",
                    probe_command(model, batch, length, inputs, conversion),
                    rejected_shape=rejection,
                )
                if not passed and rejection is None:
                    return 1
        for length in lengths:
            for batch in batches:
                name = case_id(condition, batch, length)
                reference = report["stages"].get(case_id("0", 1, length), {})
                # Exit 3 means the predictions passed but the requested profiles did not.
                if (condition, batch) != ("0", 1) and reference.get(
                    "exit_code"
                ) not in (0, 3):
                    report["stages"][name] = {
                        "verified": False,
                        "reason": "Missing FP32 batch 1 reference",
                    }
                    write_json(root / "sweep.json", report)
                    continue
                run_stage(
                    name,
                    case_command(
                        model, condition, batch, length, inputs, root, conversion
                    ),
                )
    return 0 if all(stage["verified"] for stage in report["stages"].values()) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("clipreid", "gvhmr"), required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_queue()
    return run_sweep(
        args.model, args.inputs.resolve(strict=True), args.output.resolve()
    )


if __name__ == "__main__":
    sys.exit(main())
