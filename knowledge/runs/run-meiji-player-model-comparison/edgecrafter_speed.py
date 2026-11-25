"""Measure one EdgeCrafter case on the shared #2496 full-image inputs."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import TypeVar

import numpy as np
import torch
from edgecrafter import (
    ASSETS,
    BACKENDS,
    SCORE_THRESHOLD,
    Frames,
    make_benchmark_adapter,
)
from edgecrafter_parity import _prediction, detection_output
from run import load_conversion, read_json

from src.submodules.models.ecpose.releases import IMAGE_SIZE, RELEASES, SOURCE_COMMIT
from src.tennis_scene.evaluation.player_pose.scoring import Predictions
from src.utils.checksum import dual_sha256
from src.utils.inference_benchmark import (
    Backend,
    Batch,
    BatchStages,
    BenchmarkAdapter,
    BenchmarkCase,
    ConditionUnavailable,
    ParityResult,
    Status,
    benchmark,
)
from src.utils.inference_benchmark_inputs import PinnedBenchmarkInputs
from src.utils.inference_parity import compare_detections
from src.utils.io import write_json_atomic

Value = TypeVar("Value")


def _operation(operation: Callable[[], Value]) -> Value:
    try:
        return operation()
    except (torch.OutOfMemoryError, ConditionUnavailable):
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise ConditionUnavailable(
            Status.UNAVAILABLE, f"{type(error).__name__}: {error}"
        ) from error


def _adapter_with_receipt(
    adapter: BenchmarkAdapter[Frames, list[Predictions]],
) -> BenchmarkAdapter[Frames, list[Predictions]]:
    def load() -> Callable[[Frames, BatchStages], list[Predictions]]:
        infer = _operation(adapter.load)

        def run(rows: Frames, stages: BatchStages) -> list[Predictions]:
            return _operation(lambda: infer(rows, stages))

        return run

    return BenchmarkAdapter(adapter.backend, load, adapter.weights, adapter.engines)


def read_pt32_reference(
    directory: Path, *, case: BenchmarkCase, has_pose: bool
) -> dict[str, Predictions]:
    report_path = directory / "speed-b1.json"
    predictions_path = directory / "speed-b1-predictions.json"
    if not report_path.is_file() or not predictions_path.is_file():
        raise ConditionUnavailable(
            Status.PARITY_UNAVAILABLE,
            f"PT32 batch 1 speed reference is absent at {directory}",
        )
    report = read_json(report_path)
    saved_case = report["case"]
    settings = saved_case["settings"]
    settings_keys = (
        "official_source_commit",
        "weight_sha256",
        "dataset_id",
        "timing_protocol_sha256",
        "image_size_hw",
        "pose_box_margin",
        "person_score_exclusive",
    )
    if (
        report["status"] != "ok"
        or saved_case["name"] != case.name
        or saved_case["backend"] != Backend.PT32.value
        or saved_case["batch_size"] != 1
        or saved_case["input_manifest_sha256"] != case.input_manifest_sha256
        or any(settings[key] != case.settings[key] for key in settings_keys)
    ):
        raise ConditionUnavailable(
            Status.PARITY_UNAVAILABLE,
            f"PT32 batch 1 speed reference differs from this input or model: {directory}",
        )
    predictions = read_json(predictions_path)
    if predictions["input_manifest_sha256"] != case.input_manifest_sha256:
        raise ConditionUnavailable(
            Status.PARITY_UNAVAILABLE,
            f"PT32 prediction manifest differs: {predictions_path}",
        )
    if report["parity"]["metrics"]["predictions_sha256"] != dual_sha256(
        predictions_path
    ):
        raise ConditionUnavailable(
            Status.PARITY_UNAVAILABLE,
            f"PT32 prediction checksum differs: {predictions_path}",
        )
    return {
        key: _prediction(value, has_pose) for key, value in predictions["rows"].items()
    }


def compare_speed_rows(
    reference: Mapping[str, Predictions],
    candidate: Mapping[str, Predictions],
    *,
    has_pose: bool,
) -> ParityResult:
    return compare_detections(
        {key: detection_output(value) for key, value in reference.items()},
        {key: detection_output(value) for key, value in candidate.items()},
        with_pose=has_pose,
    )


def one_case(args: argparse.Namespace) -> Status:
    backend = BACKENDS[args.backend]
    release = RELEASES[args.model]
    fixed = PinnedBenchmarkInputs(
        args.inputs,
        args.timing_protocol,
        manifest_sha256=args.inputs_sha256,
        timing_sha256=args.timing_protocol_sha256,
    )
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Speed measurement requires a CUDA device")
    schedule = fixed.schedule("frames", args.batch_size)
    images = {}
    for row in fixed.rows("frames"):
        path = fixed.verified_asset(row)
        frame = np.load(path, allow_pickle=False)
        if frame.dtype != np.uint8 or list(frame.shape) != row["shape"]:
            raise ValueError(f"Unexpected native BGR frame shape or dtype: {path}")
        images[row["id"]] = frame
    batches: list[Batch[Frames]] = [
        Batch(ids, tuple(images[key] for key in ids)) for ids in schedule.batch_ids
    ]
    schedule.require_batches([batch.ids for batch in batches])
    conversion = tuple(load_conversion(path) for path in args.conversion)
    case = BenchmarkCase(
        args.model,
        backend,
        args.batch_size,
        "image",
        fixed.manifest_sha256,
        {
            "dataset_id": fixed.manifest["id"],
            "timing_protocol_sha256": fixed.timing_sha256,
            "timed_input_ids_sha256": schedule.timed_ids_sha256,
            "input_decode": "predecoded BGR",
            "cuda_stream": "non-default",
            "host_threads": torch.get_num_threads(),
            "image_size_hw": list(IMAGE_SIZE),
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "official_source_commit": SOURCE_COMMIT,
            "weight_sha256": release.sha256,
            "person_score_exclusive": SCORE_THRESHOLD,
            "pose_box_margin": args.pose_box_margin,
            "postprocess": "torch.float32",
            "flip_test": False,
            "crops": 0,
            "pytorch_precision": "CUDA autocast FP16"
            if backend is Backend.PT16
            else None,
            "engine_profile": None
            if args.engine is None or not (args.engine / "engine.json").is_file()
            else read_json(args.engine / "engine.json")["profiles"],
        },
    )
    adapter = _adapter_with_receipt(
        make_benchmark_adapter(
            args.model,
            args.assets,
            backend,
            device,
            pose_box_margin=args.pose_box_margin,
            engine=args.engine,
        )
    )

    def check(
        predictions: Iterator[tuple[Batch[Frames], list[Predictions]]],
    ) -> ParityResult:
        actual: dict[str, Predictions] = {}
        for batch, values in predictions:
            for key, value in zip(batch.ids, values, strict=True):
                if key in actual:
                    raise ValueError(f"Duplicate speed input ID: {key}")
                actual[key] = value
        expected_ids = tuple(key for batch in batches for key in batch.ids)
        if tuple(actual) != expected_ids:
            raise ValueError("Speed predictions do not cover the fixed batch order")
        path = args.output / f"speed-b{args.batch_size}-predictions.json"
        write_json_atomic(
            path, {"input_manifest_sha256": fixed.manifest_sha256, "rows": actual}
        )
        reference = (
            actual
            if backend is Backend.PT32 and args.batch_size == 1
            else read_pt32_reference(
                args.reference, case=case, has_pose=release.has_pose
            )
        )
        parity = compare_speed_rows(reference, actual, has_pose=release.has_pose)
        return ParityResult(
            parity.passed,
            {**parity.metrics, "predictions_sha256": dual_sha256(path)},
            parity.reason,
            parity.criteria,
        )

    report = benchmark(
        case,
        batches,
        {backend: adapter},
        device=device,
        conversion=conversion,
        check_parity=check,
        protocol=schedule.protocol,
        dedicated_stream=True,
    )
    path = args.output / f"speed-b{args.batch_size}.json"
    write_json_atomic(path, report.to_dict())
    print(
        f"{args.model} {backend} batch {args.batch_size} {report.status}. Receipt {path}"
    )
    if report.failure is not None:
        print(report.failure.message)
    return report.status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=RELEASES, required=True)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--timing-protocol", type=Path, required=True)
    parser.add_argument("--timing-protocol-sha256", required=True)
    parser.add_argument("--assets", type=Path, default=ASSETS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--conversion", type=Path, action="append", default=[])
    parser.add_argument("--pose-box-margin", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    one_case(parser.parse_args())


if __name__ == "__main__":
    main()
