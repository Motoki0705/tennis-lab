"""Measure one EdgeCrafter case on the shared #2496 full-image inputs."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import TypeVar

import torch
from edgecrafter import (
    ASSETS,
    BACKENDS,
    SCORE_THRESHOLD,
    Frames,
    make_benchmark_adapter,
)
from edgecrafter_parity import _prediction, prediction_parity
from run import SWEEPS, inputs, load_conversion, read_json
from speed_inputs import InputSet

from src.submodules.models.ecpose.releases import IMAGE_SIZE, RELEASES, SOURCE_COMMIT
from src.tennis_scene.evaluation.player_pose.scoring import Predictions
from src.utils.checksum import dual_sha256
from src.utils.inference_benchmark import (
    Backend,
    Batch,
    BatchStages,
    BenchmarkAdapter,
    BenchmarkCase,
    BenchmarkProtocol,
    ConditionUnavailable,
    ParityResult,
    Status,
    benchmark,
)
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
    if reference.keys() != candidate.keys():
        return ParityResult(
            False,
            {"reference_ids": len(reference), "candidate_ids": len(candidate)},
            "Speed input IDs differ from PT32",
        )
    return prediction_parity(
        ((reference[key], value) for key, value in candidate.items()), has_pose=has_pose
    )


def one_case(args: argparse.Namespace) -> Status:
    backend = BACKENDS[args.backend]
    release = RELEASES[args.model]
    fixed = InputSet(args.inputs)
    if fixed.sha256 != args.inputs_sha256:
        raise ValueError(
            f"Expected fixed speed manifest SHA-256 {args.inputs_sha256}, got {fixed.sha256}"
        )
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Speed measurement requires a CUDA device")
    protocol = BenchmarkProtocol(batch_sizes=SWEEPS["dino"])
    batches: list[Batch[Frames]] = inputs(fixed, "dino", args.batch_size)
    conversion = tuple(load_conversion(path) for path in args.conversion)
    case = BenchmarkCase(
        args.model,
        backend,
        args.batch_size,
        "image",
        fixed.sha256,
        {
            "dataset_id": fixed.manifest["id"],
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
        write_json_atomic(path, {"input_manifest_sha256": fixed.sha256, "rows": actual})
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
        )

    with torch.cuda.stream(torch.cuda.Stream(device=device)):
        report = benchmark(
            case,
            batches,
            {backend: adapter},
            device=device,
            conversion=conversion,
            check_parity=check,
            protocol=protocol,
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
    parser.add_argument("--batch-size", choices=SWEEPS["dino"], type=int, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--inputs-sha256", required=True)
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
