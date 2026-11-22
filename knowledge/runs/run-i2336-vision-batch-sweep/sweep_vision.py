"""Run ViTPose-H, HMR2 or DINO timing and profiling in the training queue."""

import argparse
import os
from pathlib import Path

from benchmark import NSYS, write_json
from benchmark import run_stage as run_measurement
from convert import require_queue
from inputs import sha256_file
from measure_dino import INPUT_NAME
from profiles import capture
from sweep import case_id, python, run_stage
from vision_inputs import (
    BATCHES,
    BUCKET,
    CONDITIONS,
    DINO,
    MODELS,
    OPT_BATCH,
    PinnedInputs,
    dino_method,
    load_conversion,
    read,
)

HERE = Path(__file__).resolve().parent
DINO_BUNDLE = HERE.parent / "run-i2304-dino-tensorrt"
MEASURE = HERE / "measure_vision.py"
NCU = Path("/usr/local/cuda/bin/ncu")


def measurement_command(data, condition, batch, output, mode, conversion=None):
    return python(
        MEASURE,
        "--model",
        data.model,
        "--condition",
        condition,
        "--batch-size",
        batch,
        "--inputs",
        data.directory,
        "--output",
        output,
        "--mode",
        mode,
        *(["--conversion", conversion] if conversion else []),
    )


def run_case(data, condition, batch, output, reference, conversion):
    output.mkdir(parents=True)
    engine = {}
    if conversion:
        plan, receipt, digest = load_conversion(data, condition, conversion)
        engine = {
            "conversion": str(conversion),
            "engine_sha256": digest,
            "input_profile": receipt["input_profile"],
            "msda_method": receipt.get("method"),
            "plan": plan.name,
        }
    run_measurement(
        measurement_command(data, condition, batch, output, "measure", conversion),
        output / "measure.log",
        timeout=1200,
    )
    measurement = read(output / "measurement.json")
    if measurement["batch_shape"] != [batch, *data.input_shape]:
        raise ValueError("Measured batch shape differs from the request")
    actual = data.predictions(output)
    prediction_digest = sha256_file(output / data.prediction_name)
    baseline, baseline_digest = actual, prediction_digest
    if (condition, batch) != ("0", 1):
        source = read(reference / "result.json")
        baseline_digest = sha256_file(reference / data.prediction_name)
        if (
            source["identity"] != data.identity
            or (source["condition"], source["batch_size"]) != ("0", 1)
            or source["prediction_sha256"] != baseline_digest
        ):
            raise ValueError("FP32 B=1 reference identity or predictions changed")
        baseline = data.predictions(reference)
    report = {
        "identity": data.identity,
        "condition": condition,
        "batch_size": batch,
        "queue_job": os.environ["TENNIS_RUN_ID"],
        "representative_addresses": data.representative(batch),
        "prediction_sha256": prediction_digest,
        "fp32_reference_sha256": baseline_digest,
        "accuracy": data.compare(actual, baseline),
        **engine,
    }
    nsys = capture(
        "nsys",
        str(NSYS),
        measurement_command(data, condition, batch, output, "profile", conversion),
        output,
    )
    report["nsys"] = nsys
    write_json(output / "result.json", report)
    return 0 if nsys["status"] == "measured" else 3


def run_ncu(data, args):
    source = args.profile_from.resolve(strict=True)
    prior = read(source / "result.json")
    condition, batch = prior["condition"], prior["batch_size"]
    if (
        prior["identity"] != data.identity
        or condition not in CONDITIONS[data.model]
        or batch not in BATCHES[data.model]
        or prior["representative_addresses"] != data.representative(batch)
        or prior["prediction_sha256"] != sha256_file(source / data.prediction_name)
        or prior["nsys"]["status"] != "measured"
    ):
        raise ValueError("ncu requires the corresponding pinned, measured nsys case")
    if batch != OPT_BATCH[data.model] and not (data.model == DINO and batch in {1, 2}):
        raise ValueError("ncu requires the representative batch size")
    conversion = None
    if "conversion" in prior:
        conversion = Path(prior["conversion"])
        _, receipt, digest = load_conversion(data, condition, conversion)
        if (
            digest != prior["engine_sha256"]
            or receipt["input_profile"] != prior["input_profile"]
        ):
            raise ValueError("ncu TensorRT plan differs from the nsys case")
    options = {}
    if args.ncu_prefix:
        prefix = read(args.ncu_prefix / "ncu.json")
        if (
            prefix["measurement"] != prior
            or prefix["source"] != str(source)
            or prefix["replay_mode"] != args.ncu_replay
            or prefix["ncu"]["status"] != "unavailable"
            or "timed out after" not in prefix["ncu"].get("reason", "")
        ):
            raise ValueError("ncu prefix did not time out on the same measurement")
        options = {
            "ncu_prefix": args.ncu_prefix / "ncu.csv",
            "nsys_trace": source / "nsys.sqlite",
        }
    args.output.mkdir(parents=True)
    counters = capture(
        "ncu",
        str(NCU),
        measurement_command(data, condition, batch, args.output, "profile", conversion),
        args.output,
        ncu_timeout=args.ncu_timeout,
        ncu_replay=args.ncu_replay,
        **options,
    )
    expected = prior["nsys"]["gpu_kernels"]
    if counters["status"] == "measured" and counters["profiled_kernels"] != expected:
        counters.update(
            status="coverage_mismatch",
            reason=f"nsys={expected}, ncu={counters['profiled_kernels']}",
        )
    write_json(
        args.output / "ncu.json",
        {
            "source": str(source),
            "measurement": prior,
            "ncu": counters,
            "nsys_kernels": expected,
            "queue_job": os.environ["TENNIS_RUN_ID"],
            "timeout_seconds": args.ncu_timeout,
            "replay_mode": args.ncu_replay,
        },
    )
    return 0 if counters["status"] == "measured" else 3


def build_conversion(data, condition, root, report, args):
    conversion = root / f"engine-{condition}"
    graph = args.build_from or (root / f"graph-{condition}")
    options = ["--inputs", data.directory]
    if data.model == DINO:
        options.extend(("--bucket", BUCKET))
        method = dino_method(condition)
        converter = DINO_BUNDLE / "convert_dino.py"
        if args.build_from is None:
            exported = root / f"export-{condition}"
            if not run_stage(
                root,
                report,
                f"export-{condition}",
                python(
                    DINO_BUNDLE / "export_dino.py",
                    *options,
                    "--output",
                    exported,
                    "--method",
                    method,
                    "--dynamic-batch",
                ),
            ):
                return None
            if not run_stage(
                root,
                report,
                f"prepare-{condition}",
                python(
                    converter,
                    "prepare",
                    *options,
                    "--condition",
                    "2",
                    "--source",
                    exported,
                    "--output",
                    graph,
                ),
            ):
                return None
        options.extend(("--condition", "2"))
        if args.workspace_mib is not None:
            options.extend(("--workspace-mib", args.workspace_mib))
    else:
        converter = HERE / "convert_vit_batch.py"
        options.extend(("--model", data.model, "--condition", condition))
        if args.build_from is None and not run_stage(
            root,
            report,
            f"prepare-{condition}",
            python(
                converter,
                "prepare",
                *options,
                "--output",
                graph,
            ),
        ):
            return None
    optimum, maximum = args.batch_profile or (
        OPT_BATCH[data.model],
        max(BATCHES[data.model]),
    )
    profile = data.profile(optimum, maximum)
    if not run_stage(
        root,
        report,
        f"build-{condition}",
        python(
            converter,
            "build",
            *options,
            "--source",
            graph,
            "--output",
            conversion,
            "--batch-profile",
            optimum,
            maximum,
            *(
                ["--builder-optimization-level", args.builder_optimization_level]
                if args.builder_optimization_level is not None
                else []
            ),
        ),
    ):
        return None
    _, receipt, _ = load_conversion(data, condition, conversion)
    if receipt["input_profile"] != profile:
        raise ValueError("Built plan uses a different input profile")
    return conversion


def run_sweep(data, args):
    root = args.output
    (root / "logs").mkdir(parents=True)
    report = {
        "model": data.model,
        "identity": data.identity,
        "queue_job": os.environ["TENNIS_RUN_ID"],
        "stages": {},
    }
    reference = args.reference or root / "c0-b1"
    batches = sorted(set(args.batch or BATCHES[data.model]))
    for condition in CONDITIONS[data.model]:
        if args.condition and condition not in args.condition:
            continue
        conversion = None
        if condition.startswith(("2", "3")):
            conversion = args.reuse_conversion or build_conversion(
                data, condition, root, report, args
            )
            if conversion is None:
                continue
            _, receipt, _ = load_conversion(data, condition, conversion)
            maximum = receipt["input_profile"][INPUT_NAME][-1][0] // data.multiplier
            rejections = []
            for batch in (0, maximum + 1):
                shape = (batch * data.multiplier, *data.input_shape)
                rejections.append(
                    run_stage(
                        root,
                        report,
                        f"reject-{case_id(condition, batch, None)}",
                        measurement_command(
                            data, condition, batch, root, "reject", conversion
                        ),
                        rejected_shape=f"{INPUT_NAME}={shape}",
                    )
                )
            if not all(rejections):
                continue
        for batch in batches:
            name = case_id(condition, batch, None)
            if (condition, batch) != ("0", 1) and not (
                reference / "result.json"
            ).is_file():
                report["stages"][name] = {
                    "verified": False,
                    "reason": "Missing FP32 B=1 result",
                }
                write_json(root / "sweep.json", report)
                continue
            command = python(
                __file__,
                "--model",
                data.model,
                "--inputs",
                data.directory,
                "--output",
                root / name,
                "--condition",
                condition,
                "--case",
                batch,
                "--reference",
                reference,
                *(["--reuse-conversion", conversion] if conversion else []),
            )
            run_stage(root, report, name, command)
    return 0 if all(stage["verified"] for stage in report["stages"].values()) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", action="append")
    parser.add_argument("--batch", action="append", type=int)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--case", type=int)
    parser.add_argument("--profile-from", type=Path)
    parser.add_argument("--ncu-timeout", default=2400, type=int)
    parser.add_argument(
        "--ncu-replay", choices=("kernel", "application"), default="kernel"
    )
    parser.add_argument("--ncu-prefix", type=Path)
    reusable = parser.add_mutually_exclusive_group()
    reusable.add_argument("--reuse-conversion", type=Path)
    reusable.add_argument("--build-from", type=Path)
    parser.add_argument("--batch-profile", nargs=2, type=int, metavar=("OPT", "MAX"))
    parser.add_argument("--builder-optimization-level", choices=range(6), type=int)
    parser.add_argument("--workspace-mib", type=int)
    args = parser.parse_args()
    if set(args.condition or ()) - set(CONDITIONS[args.model]):
        parser.error("Condition unavailable for this model")
    if set(args.batch or ()) - set(BATCHES[args.model]):
        parser.error("Specify a batch from the pinned sweep")
    if args.case is not None and (
        args.case not in BATCHES[args.model]
        or len(args.condition or ()) != 1
        or args.reference is None
    ):
        parser.error("Case requires one condition, a sweep batch and a reference path")
    if (args.reuse_conversion or args.build_from) and (
        len(args.condition or ()) != 1 or not args.condition[0].startswith(("2", "3"))
    ):
        parser.error("Reusing an artifact requires one TensorRT condition")
    if args.ncu_timeout <= 0 or (args.ncu_prefix and not args.profile_from):
        parser.error("Invalid ncu retry options")
    if args.workspace_mib is not None and (
        args.workspace_mib <= 0
        or args.model != DINO
        or args.condition != ["2"]
        or args.case is not None
        or args.profile_from is not None
        or args.reuse_conversion is not None
    ):
        parser.error("An explicit workspace limit applies to the DINO GridSample build")
    require_queue()
    args.inputs = args.inputs.resolve(strict=True)
    args.output = args.output.resolve()
    data = PinnedInputs(args.model, args.inputs, verify=args.case is None)
    if args.case is not None:
        return run_case(
            data,
            args.condition[0],
            args.case,
            args.output,
            args.reference,
            args.reuse_conversion,
        )
    (Path(os.environ["TENNIS_REPRO_DIR"]) / "output_dir.txt").write_text(
        f"{args.output}\n"
    )
    if args.profile_from:
        return run_ncu(data, args)
    return run_sweep(data, args)


if __name__ == "__main__":
    raise SystemExit(main())
