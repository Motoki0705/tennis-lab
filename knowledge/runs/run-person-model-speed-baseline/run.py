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

import cv2
import torch
from parity import compare, serialize
from speed_inputs import (
    BENCHMARK,
    BODY_LENGTH,
    CLIPS,
    InputSet,
    publish,
    read_source,
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
from src.utils.checksum import dual_sha256
from src.utils.inference_benchmark import (
    Backend,
    Batch,
    BenchmarkAdapter,
    BenchmarkCase,
    BenchmarkProtocol,
    ConditionUnavailable,
    Conversion,
    ConversionResult,
    Failure,
    Status,
    benchmark,
    measure_conversion,
)
from src.utils.io import write_json_atomic
from src.utils.models.tensorrt_artifacts import (
    EXPORT_MANIFEST,
    PLAN_FILE,
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
        report_sha = dual_sha256(manifest)
        stage.rename(destination)
    print(
        f"Prepared {len(frames)} frames, {len(crops)} crops and {len(bodies)} sequences at {destination}"
    )
    print(f"Input manifest SHA-256 {report_sha}")


def build_pose_b1(args) -> int:
    source = engine_paths(args.trt_root)["vitpose"]
    directory = args.output / "engines" / "vitpose-b1"
    receipt = args.output / "vitpose-b1-build.json"
    if receipt.exists():
        return int(read_json(receipt)["failure"] is not None)
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


def inputs(fixed: InputSet, workload: str, batch_size: int) -> list[Batch]:
    if workload in ("dino", "dino-vitpose"):
        rows = [(row["id"], fixed.frame(row["id"])) for row in fixed.rows("frames")]
    elif workload == "body":
        rows = [(row["id"], fixed.body(row)) for row in fixed.rows("body_sequences")]
    else:
        rows = [
            (row["id"], (row["frame_id"], fixed.frame(row["frame_id"]), row["xys"]))
            for row in fixed.rows("crops")[: fixed.manifest["pose_sweep_crop_count"]]
        ]
    if len(rows) % batch_size:
        raise ValueError(f"Non-divisible fixed input set for {workload}/B={batch_size}")
    batches = []
    for start in range(0, len(rows), batch_size):
        ids, data = zip(*rows[start : start + batch_size], strict=True)
        count = len(data) * BODY_LENGTH if workload == "body" else len(data)
        batches.append(Batch(ids, data, items=count, processed_items=count))
    return batches


def result_directory(output, workload, backend, batch_size):
    return output / "results" / workload / f"{backend}-b{batch_size}"


def one_case(args) -> None:
    workload, mode, batch_size = args.workload, Backend(args.backend), args.batch
    if batch_size not in SWEEPS[workload]:
        raise ValueError(f"Batch {batch_size} is outside {workload}'s frozen sweep")
    config = people_config(args.external_assets)
    fixed = InputSet(args.output / "inputs")
    directory = result_directory(args.output, workload, mode, batch_size)
    directory.mkdir(parents=True, exist_ok=True)
    stages = engine_paths(args.trt_root)
    conversion = ()
    if workload == "vitpose" and mode is Backend.TRT16 and batch_size == 1:
        stages["vitpose"] = args.output / "engines/vitpose-b1"
        conversion = (load_conversion(args.output / "vitpose-b1-build.json"),)
    includes_dino = workload in ("dino", "dino-vitpose")
    includes_pose = workload in ("vitpose", "vitpose-flip", "dino-vitpose")
    flip = workload in ("vitpose-flip", "dino-vitpose")
    settings = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dataset_id": fixed.manifest["id"],
        "host_threads": torch.get_num_threads(),
        "input_decode": "predecoded BGR",
        "cuda_stream": "non-default",
    }
    selected, weights, models = [], {}, {}
    if includes_dino:
        selected.append("dino")
        weights["dino"] = config.dino_checkpoint
        if mode is not Backend.TRT16:
            weights["dino_msda_cuda"] = Path(os.environ["TENNIS_LAB_DINO_EXTENSION"])
        settings.update(
            dino_hw=[750, 1333],
            dino_confidence=config.runtime.dino_detector.confidence,
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
    settings["engine_profiles"] = {
        name: read_json(stages[name] / "engine.json")["profiles"]
        for name in selected
        if mode is Backend.TRT16 and (stages[name] / "engine.json").is_file()
    }
    if mode is Backend.PT16:
        settings["pytorch_precision"] = "CUDA autocast FP16 with FP32 parameters"

    def loader():
        if workload == "body":
            models["body"] = Body(config, mode, stages, batch_size)
            return models["body"]
        dino = Dino(config, mode, stages["dino"], batch_size) if includes_dino else None
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
            for key, value in zip(batch.ids, values, strict=True):
                if key in actual:
                    raise ValueError(f"Duplicate prediction ID: {key}")
                actual[key] = serialize(workload, value, models.get("body"))
        path = directory / "predictions.json"
        write_json_atomic(path, actual)
        if mode is Backend.PT32 and batch_size == 1:
            return compare(workload, actual, actual)
        reference = result_directory(args.output, workload, Backend.PT32, 1)
        report = reference / "report.json"
        if not report.is_file() or read_json(report)["status"] != "ok":
            raise ConditionUnavailable(
                Status.PARITY_UNAVAILABLE,
                f"PT32 B=1 reference absent or failed: {reference}",
            )
        if read_json(report)["case"]["input_manifest_sha256"] != fixed.sha256:
            raise ConditionUnavailable(
                Status.PARITY_UNAVAILABLE,
                "PT32 reference input manifest SHA-256 differs",
            )
        return compare(workload, read_json(reference / "predictions.json"), actual)

    case = BenchmarkCase(
        workload, mode, batch_size, CASE_UNITS[workload], fixed.sha256, settings
    )
    engine_files = (
        {name: stages[name] / PLAN_FILE for name in selected}
        if mode is Backend.TRT16
        else {}
    )
    adapter = BenchmarkAdapter(mode, loader, weights, engine_files)
    protocol = BenchmarkProtocol(batch_sizes=SWEEPS[workload])
    batches = inputs(fixed, workload, batch_size)
    with torch.cuda.stream(torch.cuda.Stream(device=DEVICE)):
        report = benchmark(
            case,
            batches,
            {mode: adapter},
            protocol=protocol,
            device=DEVICE,
            conversion=conversion,
            check_parity=parity,
        )
    write_json_atomic(directory / "report.json", report.to_dict())
    print(f"{workload}/{mode}/B{batch_size} {report.status}", flush=True)
    if report.failure:
        print(f"{report.failure.phase}: {report.failure.message}", flush=True)


def suite(args) -> int:
    fixed = InputSet(args.output / "inputs")
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
    ]
    if args.group == "pose":
        result = subprocess.run([*prefix, "build-pose-b1"], check=False)
        if result.returncode and not (args.output / "vitpose-b1-build.json").is_file():
            print(
                "ViTPose build ended before publishing a conversion receipt",
                file=sys.stderr,
                flush=True,
            )
    for workload in GROUPS[args.group]:
        for backend in Backend:
            for batch_size in SWEEPS[workload]:
                directory = result_directory(args.output, workload, backend, batch_size)
                directory.mkdir(parents=True, exist_ok=True)
                report = directory / "report.json"
                command = [*prefix, "case", workload, backend.value, str(batch_size)]
                if not report.exists():
                    with (directory / "case.log").open("w") as log:
                        code = subprocess.run(
                            command, stdout=log, stderr=subprocess.STDOUT, check=False
                        ).returncode
                else:
                    if (
                        read_json(report)["case"]["input_manifest_sha256"]
                        != fixed.sha256
                    ):
                        raise ValueError(
                            f"Existing measurement refers to different input bytes: {report}"
                        )
                    code = 0
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
    args = parser.parse_args()
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
