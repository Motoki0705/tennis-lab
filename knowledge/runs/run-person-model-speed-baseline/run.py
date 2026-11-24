"""Run the fixed Meiji 4cam current-model inference comparison."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import cv2
import torch
from parity import compare, serialize
from speed_inputs import (
    BENCHMARK,
    BODY_LENGTH,
    CLIPS,
    INPUT_MANIFEST_SHA256,
    TIMING_PROTOCOL_SHA256,
    WORKLOAD_INDEX,
    InputSet,
    publish,
    read_source,
    require_sha,
    write_crops,
    write_frames,
)
from speed_models import (
    DEVICE,
    HMR2_CROP_BATCH,
    POSE_CROP_BATCH,
    Body,
    Dino,
    Pose,
    classify_startup,
    dino_eager,
    engine_paths,
    people_config,
    pose_groups,
    two_stage,
)

from src.submodules.models import PersonDetectionRequest
from src.tennis_scene.pipeline.model_resources import (
    dino_tensorrt,
    gvhmr_tensorrt,
    hmr2_tensorrt,
    vitpose_tensorrt,
)
from src.utils.checksum import FileIntegrityError, dual_sha256
from src.utils.inference_benchmark import (
    Backend,
    Batch,
    BenchmarkAdapter,
    BenchmarkCase,
    ConditionUnavailable,
    Conversion,
    ConversionResult,
    Failure,
    ParityResult,
    Status,
    benchmark,
    measure_conversion,
)
from src.utils.inference_benchmark_inputs import TimingSchedule, read_pinned_json
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

SWEEPS = {
    "dino": (1, 2),
    "vitpose-flip": (1, 4, 8),
    "vitpose": (1, 4, 8),
    "dino-vitpose": (1, 2),
    "body": (1, 2, 4),
}
GROUPS = {
    "detection": ("dino", "dino-vitpose"),
    "pose": ("vitpose-flip", "vitpose"),
    "body": ("body",),
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
    config = people_config(args.external_assets)
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


def prepare(args) -> None:
    config = people_config(args.external_assets)
    source = args.benchmark.resolve(strict=True)
    destination = args.output / "inputs"
    if destination.exists():
        fixed = InputSet(destination)
        for name in fixed.manifest["indexes"]:
            fixed.rows(name)
        print(f"Already prepared {fixed.manifest['id']} {fixed.sha256}")
        return
    clips = read_source(source)
    with TemporaryDirectory(prefix=".inputs-", dir=args.output) as temporary:
        stage = Path(temporary)
        frames, extra, bodies = write_frames(source, stage, clips)
        if len(frames) != len(CLIPS) * 4 * 4 or len(bodies) != len(CLIPS) * 4:
            raise ValueError("Unexpected count of fixed full-frame or body rows")
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        model = dino_eager(config, Backend.PT32)
        model.load()
        crops = write_crops(
            stage,
            frames,
            clips,
            lambda image: model.predict(PersonDetectionRequest(image)),
        )
        manifest = publish(
            stage,
            frames,
            extra,
            bodies,
            crops,
            {
                "model": "dino_4scale_swin_l",
                "precision": "PT32",
                "tf32": False,
                "confidence": model.confidence,
                "short_side": model.short_side,
                "max_long_side": model.max_long_side,
                "checkpoint_sha256": dual_sha256(model.checkpoint),
                "extension_sha256": dual_sha256(
                    Path(os.environ["TENNIS_LAB_DINO_EXTENSION"])
                ),
                "external_source_sha": subprocess.check_output(
                    ["git", "-C", str(model.repository), "rev-parse", "HEAD"], text=True
                ).strip(),
            },
        )
        require_sha(manifest, INPUT_MANIFEST_SHA256)
        stage.rename(destination)
    print(
        f"Prepared {len(frames)} frames, {len(crops)} crops and {len(bodies)} sequences at {destination}"
    )
    print(f"Input manifest SHA-256 {INPUT_MANIFEST_SHA256}")


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


def reference_predictions(
    directory: Path, case: BenchmarkCase, weights: dict[str, Path]
):
    try:
        report = read_json(directory / "report.json")
        referenced = report["case"]
        if report["status"] != "ok" or (
            referenced["name"],
            referenced["backend"],
            referenced["batch_size"],
            referenced["input_manifest_sha256"],
            referenced["settings"]["reference_identity"],
        ) != (
            case.name,
            "PT32",
            1,
            case.input_manifest_sha256,
            case.settings["reference_identity"],
        ):
            raise ValueError("PT32 reference identity or status differs")
        if report["weights"].keys() != weights.keys():
            raise ValueError("PT32 reference weight inventory differs")
        for name, path in weights.items():
            pinned = report["weights"][name]
            if path.resolve() != Path(pinned["path"]).resolve():
                raise ValueError(f"PT32 reference asset path differs for {name}")
            if not path.is_file():
                raise FileNotFoundError(path)
            require_sha(path, pinned["sha256"])
        return read_pinned_json(
            directory / "predictions.json", report["predictions_sha256"]
        )
    except (OSError, ValueError, KeyError, FileIntegrityError) as error:
        raise ConditionUnavailable(
            Status.PARITY_UNAVAILABLE, f"PT32 B=1 reference at {directory}: {error}"
        ) from error


def one_case(args) -> None:
    workload, mode, batch_size = args.workload, Backend(args.backend), args.batch
    if batch_size not in SWEEPS[workload]:
        raise ValueError(f"Batch {batch_size} is outside {workload}'s frozen sweep")
    invocation = check_invocation(args)
    config = people_config(args.external_assets)
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
    includes_dino = workload in ("dino", "dino-vitpose")
    includes_pose = workload in ("vitpose", "vitpose-flip", "dino-vitpose")
    flip = workload in ("vitpose-flip", "dino-vitpose")
    settings = {
        "source_commit": invocation["source_commit"],
        "dataset_id": fixed.manifest["id"],
        "host_threads": torch.get_num_threads(),
        "input_decode": "predecoded BGR",
        "cuda_stream": "non-default",
        "timing_protocol_sha256": fixed.pins.timing_sha256,
        "timed_input_ids_sha256": schedule.timed_ids_sha256,
    }
    selected, weights, models = [], {}, {}
    if includes_dino:
        selected.append("dino")
        weights["dino"] = config.dino_checkpoint
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
    if includes_pose:
        selected.append("vitpose")
        weights["vitpose"] = config.vitpose_checkpoint
        settings.update(
            vitpose_hw=[256, 192],
            flip_test=flip,
            pose_crop_batch=POSE_CROP_BATCH if includes_dino else batch_size,
            pose_head=json_value(config.runtime.vitpose.head),
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
    if mode is Backend.PT16:
        settings["pytorch_precision"] = "CUDA autocast FP16 with FP32 parameters"
    settings["reference_identity"] = {
        "suite_id": args.run_id,
        "commit": invocation["source_commit"],
        "timing": fixed.pins.timing_sha256,
        "contract": {
            key: settings[key]
            for key in (
                "dino_hw",
                "dino_confidence",
                "dino_resize",
                "vitpose_hw",
                "flip_test",
                "pose_head",
                "body_window_frames",
                "hmr2_crop_batch",
                "gvhmr_decode_static_ik",
            )
            if key in settings
        },
        "composed_pose_crop_batch": POSE_CROP_BATCH
        if includes_dino and includes_pose
        else None,
    }
    reference_weights = dict(weights)
    if includes_dino:
        extension = invocation["dino_extension"]
        reference_weights["dino_msda_cuda"] = (
            Path(extension) if extension else Path("<missing DINO extension>")
        )
        if mode is not Backend.TRT16:
            weights["dino_msda_cuda"] = reference_weights["dino_msda_cuda"]

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
                models["body"] = Body(config, mode, stages, batch_size)
                return models["body"]
            dino = (
                Dino(config, mode, stages["dino"], batch_size)
                if includes_dino
                else None
            )
            pose = (
                Pose(
                    config,
                    mode,
                    stages["vitpose"],
                    flip=flip,
                    batch=POSE_CROP_BATCH if includes_dino else batch_size,
                )
                if includes_pose
                else None
            )
        if workload == "dino":
            return dino
        if workload == "dino-vitpose":
            return lambda images, clock: two_stage(dino, pose, images, clock)
        return lambda rows, clock: pose(pose_groups(rows), clock)

    def parity(predictions):
        actual = {}
        for batch, values in predictions:
            if len(values) != len(batch.ids):
                return ParityResult(
                    False, {"invalid": True}, f"Output count differs for {batch.ids}"
                )
            for key, data, value in zip(batch.ids, batch.data, values, strict=True):
                if key in actual:
                    raise ValueError(f"Duplicate prediction ID: {key}")
                actual[key] = serialize(
                    workload,
                    value,
                    models.get("body"),
                    frame_ids=data["source_frames"] if workload == "body" else None,
                )
        path = directory / "predictions.json"
        write_json_atomic(path, actual)
        reference = actual
        if mode is not Backend.PT32 or batch_size != 1:
            reference = reference_predictions(
                result_directory(args.output, workload, Backend.PT32, 1),
                case,
                reference_weights,
            )
        try:
            return compare(workload, reference, actual)
        except (ValueError, TypeError, KeyError) as error:
            return ParityResult(
                False, {"invalid": True}, f"Invalid {workload} output: {error}"
            )

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
        check_parity=parity,
    )
    result = report.to_dict()
    predicted = directory / "predictions.json"
    if predicted.is_file():
        result["predictions_sha256"] = dual_sha256(predicted)
    write_json_atomic(directory / "report.json", result)
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
        for backend in Backend:
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
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument(
        "--pose-engine-run",
        type=Path,
        help="Reuse the verified noflip B=1 plan from this run output",
    )
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("inspect")
    actions.add_parser("prepare")
    actions.add_parser("build-pose-b1")
    sweep = actions.add_parser("suite")
    sweep.add_argument("group", choices=GROUPS)
    case = actions.add_parser("case")
    case.add_argument("workload", choices=SWEEPS)
    case.add_argument("backend", choices=[mode.value for mode in Backend])
    case.add_argument("batch", type=int)
    case.add_argument("run_id")
    args = parser.parse_args()
    args.input_dir = args.input_dir or args.output / "inputs"
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    if args.action == "inspect":
        inspect(args)
        return 0
    location(args.output)
    if args.action == "prepare":
        prepare(args)
    elif args.action == "build-pose-b1":
        return build_pose_b1(args)
    elif args.action == "suite":
        return suite(args)
    else:
        one_case(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
