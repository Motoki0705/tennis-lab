"""Run the fixed Meiji 4cam current-model inference comparison."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from functools import partial
from pathlib import Path
from uuid import uuid4

import cv2
import torch
from speed_inputs import (
    BENCHMARK,
    BODY_LENGTH,
    INPUT_MANIFEST_SHA256,
    TIMING_PROTOCOL_SHA256,
    WORKLOAD_INDEX,
    InputSet,
    read_source,
)
from speed_models import (
    DEVICE,
    HMR2_CROP_BATCH,
    Body,
    Dino,
    Pose,
    classify_startup,
    engine_paths,
    pipeline_config,
    pose_groups,
    two_stage,
)

from src.tennis_scene.pipeline.model_resources import (
    dino_tensorrt,
    gvhmr_tensorrt,
    hmr2_tensorrt,
    vitpose_tensorrt,
)
from src.utils.inference_benchmark import (
    Backend,
    Batch,
    BenchmarkAdapter,
    BenchmarkCase,
    Conversion,
    ConversionResult,
    Failure,
    Status,
    benchmark,
    measure_conversion,
)
from src.utils.inference_benchmark_inputs import TimingSchedule
from src.utils.io import json_value, write_json_atomic
from src.utils.models.tensorrt_artifacts import (
    ENGINE_MANIFEST,
    EXPORT_MANIFEST,
    PLAN_FILE,
    EngineManifest,
    ExportManifest,
    InputProfile,
)
from src.utils.models.tensorrt_build import build_bundle

BACKENDS = (Backend.PT16, Backend.TRT16)
PARITY_NOT_EVALUATED_REASON = "未判定（PT32計測はスコープ外、ユーザー判断 2026-11-25）"
JOINT_POSE_CROP_BATCH = 8
SWEEPS = {
    "dino": (1, 2),
    "dino-vitpose": (1, 2),
    "vitpose-flip": (1, 4, 8),
    "vitpose": (1, 4, 8),
    "body": (1, 2, 4),
}
GROUPS = {
    "detection": ("dino",),
    "pose": ("vitpose-flip", "vitpose"),
    "body": ("body",),
    "joint": ("dino-vitpose",),
}
CASE_UNITS = {
    "dino": "image",
    "dino-vitpose": "image",
    "vitpose-flip": "person-crop",
    "vitpose": "person-crop",
    "body": "person-frame",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def location(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    repro = os.environ.get("TENNIS_REPRO_DIR")
    if repro:
        (Path(repro) / "output_dir.txt").write_text(str(output.resolve()) + "\n")


def invocation_settings(args):
    return {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "input_dir": str(args.input_dir.resolve()),
        "input_manifest_sha256": INPUT_MANIFEST_SHA256,
        "timing_protocol_sha256": TIMING_PROTOCOL_SHA256,
        "parity_not_evaluated_reason": PARITY_NOT_EVALUATED_REASON,
        "trt_root": str(args.trt_root.resolve()),
        "external_assets": str(args.external_assets.resolve()),
        "dino_extension": os.environ.get("TENNIS_LAB_DINO_EXTENSION"),
        "pose_engine_run": None
        if args.pose_engine_run is None
        else str(args.pose_engine_run.resolve()),
    }


def start_suite(args):
    for workload in GROUPS[args.group]:
        if (args.output / "results" / workload).exists():
            raise FileExistsError(f"Use a fresh output for {args.group}: {args.output}")
    settings = invocation_settings(args)
    run_id = uuid4().hex
    with (args.output / f"invocation-{args.group}.json").open(
        "x", encoding="utf-8"
    ) as file:
        json.dump({"id": run_id, "settings": settings}, file, sort_keys=True)
    return run_id


def check_invocation(args):
    group = next(group for group, names in GROUPS.items() if args.workload in names)
    saved = read_json(args.output / f"invocation-{group}.json")
    if saved != {"id": args.run_id, "settings": invocation_settings(args)}:
        raise ValueError(f"Case differs from the fresh {group} suite invocation")
    return saved["settings"]


def inspect(args) -> None:
    config = pipeline_config(args.external_assets).people
    for clip in read_source(args.benchmark):
        print(
            clip.clip_id,
            "image times:",
            clip.image_frames,
            "body rows:",
            len(clip.body_windows),
        )
    specs = {
        "dino": dino_tensorrt(config, 750, 1333),
        "vitpose": vitpose_tensorrt(config),
        "hmr2": hmr2_tensorrt(config),
        "gvhmr": gvhmr_tensorrt(config),
    }
    for name, directory in engine_paths(args.trt_root).items():
        exported = ExportManifest.model_validate_json(
            (directory / EXPORT_MANIFEST).read_bytes()
        )
        profiles = read_json(directory / "engine.json")["profiles"]
        spec = specs[name]
        if (exported.model, exported.settings, exported.inputs) != (
            spec.model,
            spec.settings,
            spec.inputs,
        ):
            raise ValueError(
                f"Model settings differ from the engine export: {directory}"
            )
        if not (directory / PLAN_FILE).is_file() or any(
            not path.is_file() for path in spec.assets.values()
        ):
            raise FileNotFoundError(
                f"Missing engine plan or source checkpoint: {directory}"
            )
        print(name, exported.model, profiles)
    print("Source clips and existing engine descriptors verified")


def build_pose_b1(args) -> int:
    source = engine_paths(args.trt_root)["vitpose"]
    directory = args.output / "engines" / "vitpose-b1"
    receipt = args.output / "vitpose-b1-build.json"
    if receipt.exists():
        raise FileExistsError(f"Use a fresh engine build output: {receipt}")
    exported = ExportManifest.model_validate_json(
        (source / EXPORT_MANIFEST).read_bytes()
    )
    directory.mkdir(parents=True, exist_ok=False)
    for name in (EXPORT_MANIFEST, *exported.onnx_files):
        shutil.copyfile(source / name, directory / name)
    profiles = {
        name: InputProfile(
            minimum=(1, *spec.shape[1:]),
            optimum=(1, *spec.shape[1:]),
            maximum=(2, *spec.shape[1:]),
        )
        for name, spec in exported.inputs.items()
    }
    result = measure_conversion(
        Conversion.ENGINE_BUILD,
        lambda: build_bundle(
            directory,
            profiles,
            device=torch.device(DEVICE),
            workspace_bytes=1536 * 2**20,
        ),
        device=DEVICE,
    )
    write_json_atomic(receipt, asdict(result))
    print(f"ViTPose internal B=1 engine build: {receipt}", flush=True)
    if result.failure:
        print(result.failure.message, file=sys.stderr, flush=True)
    return int(result.failure is not None)


def load_conversion(path: Path) -> ConversionResult:
    row = read_json(path)
    failure = row["failure"]
    return ConversionResult(
        Conversion(row["phase"]),
        row["wall_ms"],
        None
        if failure is None
        else Failure(
            Status(failure["status"]),
            failure["phase"],
            failure["message"],
        ),
    )


def inputs(fixed: InputSet, workload: str, schedule: TimingSchedule) -> list[Batch]:
    index = WORKLOAD_INDEX[workload]
    rows = {row["id"]: row for row in fixed.rows(index)}

    def payload(key):
        row = rows[key]
        if index == "frames":
            return fixed.frame(key)
        if index == "body_sequences":
            return fixed.body(row)
        return row["frame_id"], fixed.frame(row["frame_id"]), row["xys"]

    batches = []
    for ids in schedule.batch_ids:
        count = len(ids) * (BODY_LENGTH if workload == "body" else 1)
        batches.append(
            Batch(ids, tuple(map(payload, ids)), items=count, processed_items=count)
        )
    schedule.require_batches([batch.ids for batch in batches])
    return batches


def result_directory(output, workload, backend, batch_size):
    return output / "results" / workload / f"{backend}-b{batch_size}"


def one_case(args) -> None:
    workload, mode, batch_size = args.workload, Backend(args.backend), args.batch
    if mode not in BACKENDS:
        raise ValueError(f"{mode} is outside the baseline measurement scope")
    if batch_size not in SWEEPS[workload]:
        raise ValueError(f"Batch {batch_size} is outside {workload}'s frozen sweep")
    invocation = check_invocation(args)
    pipeline = pipeline_config(args.external_assets)
    config = pipeline.people
    fixed = InputSet(args.input_dir)
    schedule = fixed.pins.schedule(WORKLOAD_INDEX[workload], batch_size)
    directory = result_directory(args.output, workload, mode, batch_size)
    (directory / "case-started").touch(exist_ok=False)
    stages = engine_paths(args.trt_root)
    conversion = ()
    if workload == "vitpose" and mode is Backend.TRT16 and batch_size == 1:
        pose_run = args.pose_engine_run or args.output
        stages["vitpose"] = pose_run / "engines/vitpose-b1"
        receipt = pose_run / "vitpose-b1-build.json"
        if receipt.is_file():
            result = load_conversion(receipt)
            if args.pose_engine_run is None or result.failure is not None:
                conversion = (result,)
    joint = workload == "dino-vitpose"
    uses_dino = workload == "dino" or joint
    uses_pose = workload in {"vitpose", "vitpose-flip"} or joint
    flip = config.runtime.vitpose.flip_test if joint else workload == "vitpose-flip"
    crop_batch = JOINT_POSE_CROP_BATCH if joint else batch_size
    settings = {
        "source_commit": invocation["source_commit"],
        "dataset_id": fixed.manifest["id"],
        "host_threads": torch.get_num_threads(),
        "input_decode": "predecoded BGR",
        "cuda_stream": "non-default",
        "timing_protocol_sha256": fixed.pins.timing_sha256,
        "timed_input_ids_sha256": schedule.timed_ids_sha256,
    }
    selected: list[str] = []
    weights: dict[str, Path] = {}
    if uses_dino:
        selected.append("dino")
        weights["dino"] = config.dino_checkpoint
        if mode is Backend.PT16:
            extension = invocation["dino_extension"]
            weights["dino_msda_cuda"] = (
                Path(extension) if extension else Path("<missing DINO extension>")
            )
        settings.update(
            dino_hw=[750, 1333],
            dino_confidence=config.runtime.dino_detector.confidence,
            dino_resize=[
                config.runtime.dino_detector.short_side,
                config.runtime.dino_detector.max_long_side,
            ],
            msda="TensorRT GridSample"
            if mode is Backend.TRT16
            else "official CUDA FP32",
        )
    if workload == "body":
        selected.extend(("hmr2", "gvhmr"))
        weights.update(config.body_assets())
        settings.update(
            body_window_frames=BODY_LENGTH,
            hmr2_crop_batch=HMR2_CROP_BATCH,
            body_coordinate_system="camera",
            gvhmr_decode_static_ik="PyTorch FP32",
        )
    if uses_pose:
        selected.append("vitpose")
        weights["vitpose"] = config.vitpose_checkpoint
        settings.update(
            vitpose_hw=[256, 192],
            flip_test=flip,
            pose_crop_batch=crop_batch,
            pose_head=json_value(config.runtime.vitpose.head),
        )
    if joint:
        if pipeline.merge_duplicate_person_boxes:
            raise ValueError(
                "The joint comparison requires the original unmerged DINO rows"
            )
        settings.update(
            merge_duplicate_person_boxes=False,
            pose_box_enlarge=pipeline.tracking.features.bbox_enlarge,
            pose_person_policy="every accepted DINO row",
            pose_people_per_image_limit=None,
        )
    if mode is Backend.PT16:
        settings["pytorch_precision"] = "CUDA autocast FP16 with FP32 parameters"

    def loader():
        with classify_startup(mode):
            if mode is Backend.TRT16:
                settings["engine_profiles"] = {
                    name: EngineManifest.model_validate_json(
                        (stages[name] / ENGINE_MANIFEST).read_bytes()
                    ).model_dump(mode="json")["profiles"]
                    for name in selected
                }
            if workload == "body":
                return Body(config, mode, stages, batch_size)
            if workload == "dino":
                return Dino(config, mode, stages["dino"], batch_size)
            if joint:
                dino = Dino(config, mode, stages["dino"], batch_size)
                pose = Pose(
                    config, mode, stages["vitpose"], flip=flip, batch=crop_batch
                )
                return partial(
                    two_stage,
                    dino,
                    pose,
                    crop_batch=crop_batch,
                    features=pipeline.tracking.features,
                )
            pose = Pose(config, mode, stages["vitpose"], flip=flip, batch=crop_batch)
        return lambda rows, clock: pose(pose_groups(rows), clock)

    case = BenchmarkCase(
        workload, mode, batch_size, CASE_UNITS[workload], fixed.sha256, settings
    )
    engine_files = (
        {name: stages[name] / PLAN_FILE for name in selected}
        if mode is Backend.TRT16
        else {}
    )
    adapter = BenchmarkAdapter(mode, loader, weights, engine_files)
    report = benchmark(
        case,
        inputs(fixed, workload, schedule),
        {mode: adapter},
        protocol=schedule.protocol,
        device=DEVICE,
        dedicated_stream=True,
        conversion=conversion,
        parity_not_evaluated_reason=invocation["parity_not_evaluated_reason"],
    )
    write_json_atomic(directory / "report.json", report.to_dict())
    print(f"{workload}/{mode}/B{batch_size} {report.status}", flush=True)
    if report.failure:
        print(f"{report.failure.phase}: {report.failure.message}", flush=True)


def suite(args) -> int:
    run_id = start_suite(args)
    fixed = InputSet(args.input_dir)
    for workload in GROUPS[args.group]:
        for size in SWEEPS[workload]:
            fixed.pins.schedule(WORKLOAD_INDEX[workload], size)
    attempts = []
    prefix = [
        sys.executable,
        "-B",
        "-u",
        __file__,
        "--output",
        str(args.output),
        "--benchmark",
        str(args.benchmark),
        "--trt-root",
        str(args.trt_root),
        "--external-assets",
        str(args.external_assets),
        "--input-dir",
        str(args.input_dir),
    ]
    if args.pose_engine_run is not None:
        prefix += ["--pose-engine-run", str(args.pose_engine_run)]
    if args.group == "pose" and args.pose_engine_run is None:
        result = subprocess.run([*prefix, "build-pose-b1"], check=False)
        receipt = args.output / "vitpose-b1-build.json"
        if result.returncode and not receipt.is_file():
            failure = Failure(
                Status.CONVERSION_FAILED,
                "engine_build",
                f"Build process exited {result.returncode} before writing a receipt",
            )
            write_json_atomic(
                receipt,
                asdict(ConversionResult(Conversion.ENGINE_BUILD, None, failure)),
            )
    for workload in GROUPS[args.group]:
        for backend in BACKENDS:
            for batch_size in SWEEPS[workload]:
                directory = result_directory(args.output, workload, backend, batch_size)
                directory.mkdir(parents=True, exist_ok=False)
                report = directory / "report.json"
                command = [
                    *prefix,
                    "case",
                    workload,
                    backend.value,
                    str(batch_size),
                    run_id,
                ]
                with (directory / "case.log").open("x") as log:
                    code = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, check=False
                    ).returncode
                state = (
                    read_json(report)["status"]
                    if report.exists() and code == 0
                    else "process_failed"
                )
                attempts.append(
                    {
                        "workload": workload,
                        "backend": backend,
                        "batch": batch_size,
                        "status": state,
                        "exit_code": code,
                        "path": str(directory.relative_to(args.output)),
                    }
                )
                write_json_atomic(args.output / f"attempts-{args.group}.json", attempts)
                print(
                    f"{workload}/{backend}/B{batch_size} {state} exit={code}",
                    flush=True,
                )
    return int(any(item["status"] == "process_failed" for item in attempts))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK)
    parser.add_argument("--trt-root", required=True, type=Path)
    parser.add_argument("--external-assets", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument(
        "--pose-engine-run",
        type=Path,
        help="Reuse the verified noflip B=1 plan from this run output",
    )
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("inspect")
    actions.add_parser("build-pose-b1")
    sweep = actions.add_parser("suite")
    sweep.add_argument("group", choices=GROUPS)
    case = actions.add_parser("case")
    case.add_argument("workload", choices=SWEEPS)
    case.add_argument("backend", choices=BACKENDS)
    case.add_argument("batch", type=int)
    case.add_argument("run_id")
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    if args.action == "inspect":
        inspect(args)
        return 0
    location(args.output)
    if args.action == "build-pose-b1":
        return build_pose_b1(args)
    elif args.action == "suite":
        return suite(args)
    else:
        one_case(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
