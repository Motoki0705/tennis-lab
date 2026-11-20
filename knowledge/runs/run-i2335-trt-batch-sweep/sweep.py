"""Run the pinned CLIP-ReID or GVHMR batch sweep inside an exclusive GPU job."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from benchmark import verify_inputs, write_json
from convert import PLAN as CLIP_PLAN
from convert import TRT_PRECISIONS, require_queue
from inputs import CHECKPOINT as CLIP_CHECKPOINT
from inputs import MODEL as CLIP_MODEL
from inputs import sha256_file

RUNS = Path(__file__).resolve().parents[1]
CLIP = RUNS / "run-i2297-clipreid-tensorrt"
GVHMR = RUNS / "run-i2305-gvhmr-tensorrt"
VIT = RUNS / "run-i2303-vit-tensorrt" / "vit_models.py"
CLIP_BATCHES = (1, 4, 8, 16, 32)
CLIP_OPT = 16
GVHMR_BATCHES = (1, 2, 4)
GVHMR_OPT = 4
GVHMR_LENGTHS = (145, 177)
CLIP_RECIPES = {"3a": "attention-out-max", "3b": "linear-max"}


def python(script, *arguments):
    return [
        sys.executable,
        "-B",
        "-X",
        "faulthandler",
        str(script),
        *map(str, arguments),
    ]


def case_id(condition, batch, length):
    name = f"c{condition}-b{batch}"
    return name if length is None else f"{name}-l{length}"


def build_command(model, condition, inputs, conversion):
    if model == "clipreid":
        ptq = (
            [
                "--adapter",
                VIT,
                "--model",
                CLIP_MODEL,
                "--ptq-recipe",
                CLIP_RECIPES[condition],
                "--calibration",
                inputs / "calibration.npy",
            ]
            if condition in CLIP_RECIPES
            else []
        )
        return python(
            CLIP / "convert.py",
            "--condition",
            condition,
            "--checkpoint",
            CLIP_CHECKPOINT,
            "--output",
            conversion,
            "--batch-profile",
            CLIP_OPT,
            max(CLIP_BATCHES),
            *ptq,
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


def probe_command(model, condition, batch, length, inputs, conversion):
    if model == "clipreid":
        return python(
            CLIP / "runtime.py",
            "--condition",
            condition,
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


def case_command(
    model, condition, batch, length, inputs, output, *, source=None, ncu_timeout=None
):
    arguments = [
        "--condition",
        condition,
        "--batch-size",
        batch,
        "--inputs",
        inputs,
        "--output",
        output,
        "--profiles",
        "ncu" if source is not None else "nsys",
    ]
    if source is not None:
        arguments.extend(("--profile-from", source, "--ncu-timeout", ncu_timeout))
    elif (condition, batch) != ("0", 1):
        arguments.extend(("--reference", output.parent / case_id("0", 1, length)))
    if condition in TRT_PRECISIONS and (source is None or model == "gvhmr"):
        base = output.parent if source is None else source.parent
        arguments.extend(("--conversion", base / f"engine-{condition}"))
    if model == "clipreid":
        return python(
            CLIP / "benchmark.py",
            *arguments,
            "--model",
            CLIP_MODEL,
        )
    return python(
        GVHMR / "experiment.py",
        *arguments,
        "--length",
        length,
    )


def run_profile(model, inputs, root, source, ncu_timeout):
    if model == "clipreid":
        identity = json.loads((source / "result.json").read_text())["identity"]
        length = None
        representative = CLIP_OPT
    else:
        identity = json.loads((source / "measurement.json").read_text())
        length = identity["length"]
        representative = GVHMR_OPT
    if identity["batch_size"] != representative:
        raise ValueError(f"Expected ncu representative B={representative}: {source}")
    command = case_command(
        model,
        identity["condition"],
        representative,
        length,
        inputs,
        root,
        source=source,
        ncu_timeout=ncu_timeout,
    )
    return subprocess.run(command, check=False).returncode


def reused_clip_measurement(path, batch, tensors):
    receipt = json.loads((path / "result.json").read_text())
    identity = receipt["identity"]
    if (
        identity["model"] != CLIP_MODEL
        or identity["condition"] != "0"
        or identity["batch_size"] != batch
        or identity["tensors"] != tensors
        or receipt["accuracy"]["crops"] != 236
        or receipt["profiles"]["nsys"]["status"] != "measured"
        or receipt["features_sha256"] != sha256_file(path / "features.npy")
    ):
        raise ValueError(
            f"Reused CLIP measurement does not match its pinned case: {path}"
        )
    return {"exit_code": 0, "verified": True, "reused_measurement": str(path)}


def run_sweep(model, inputs, root, reused=()):
    reuse = {path.name: path for path in reused}
    allow_reuse = (
        {case_id("0", batch, None) for batch in CLIP_BATCHES}
        if model == "clipreid"
        else set()
    )
    if len(reuse) != len(reused) or reuse.keys() - allow_reuse:
        raise ValueError("Select unique CLIP FP32 case directories for reuse")
    tensors = verify_inputs(inputs)[0]["tensors"] if model == "clipreid" else None
    (root / "logs").mkdir(parents=True)
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

    batches, lengths = (
        (CLIP_BATCHES, (None,))
        if model == "clipreid"
        else (GVHMR_BATCHES, GVHMR_LENGTHS)
    )
    conditions = (
        ("0", "1", "2", *CLIP_RECIPES) if model == "clipreid" else ("0", "1", "2")
    )
    for condition in conditions:
        if condition in TRT_PRECISIONS:
            conversion = root / f"engine-{condition}"
            if not run_stage(
                f"build-{condition}",
                build_command(model, condition, inputs, conversion),
            ):
                continue
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
            runnable = True
            for batch, length, rejection in probes:
                passed = run_stage(
                    f"probe-{case_id(condition, batch, length)}",
                    probe_command(model, condition, batch, length, inputs, conversion),
                    rejected_shape=rejection,
                )
                if not passed and rejection is None:
                    runnable = False
                    break
            if not runnable:
                continue
        for length in lengths:
            for batch in batches:
                name = case_id(condition, batch, length)
                destination = root / name
                if name in reuse:
                    report["stages"][name] = reused_clip_measurement(
                        reuse[name], batch, tensors
                    )
                    destination.symlink_to(reuse[name], target_is_directory=True)
                    write_json(root / "sweep.json", report)
                    continue
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
                    case_command(model, condition, batch, length, inputs, destination),
                )
                if report["stages"][name]["exit_code"] == 3:
                    # Release the queue's process group before another profiler can run.
                    return 1
    return 0 if all(stage["verified"] for stage in report["stages"].values()) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("clipreid", "gvhmr"), required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-case", type=Path, action="append", default=[])
    parser.add_argument("--profile-from", type=Path)
    parser.add_argument("--ncu-timeout", type=int, default=900)
    args = parser.parse_args()
    if args.ncu_timeout <= 0 or (args.profile_from and args.reuse_case):
        parser.error("Invalid ncu timeout or reuse options")
    require_queue()
    inputs, root = args.inputs.resolve(strict=True), args.output.resolve()
    (Path(os.environ["TENNIS_REPRO_DIR"]) / "output_dir.txt").write_text(f"{root}\n")
    if args.profile_from is not None:
        # Preserve the case path used by the sweep, including its plan-relative links.
        return run_profile(
            args.model, inputs, root, args.profile_from.absolute(), args.ncu_timeout
        )
    return run_sweep(
        args.model,
        inputs,
        root,
        [case.resolve(strict=True) for case in args.reuse_case],
    )


if __name__ == "__main__":
    sys.exit(main())
