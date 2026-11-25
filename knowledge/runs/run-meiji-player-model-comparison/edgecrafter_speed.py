"""Measure one EdgeCrafter case on the shared #2496 full-image inputs."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import cv2
import numpy as np
import torch
from edgecrafter import (
    ASSETS,
    BACKENDS,
    MEASURED_BACKENDS,
    PARITY_NOT_MEASURED,
    PARITY_NOT_MEASURED_REASON,
    SCORE_THRESHOLD,
    Frames,
    make_benchmark_adapter,
)
from run import load_conversion, read_json

from src.submodules.models.ecpose.releases import IMAGE_SIZE, RELEASES, SOURCE_COMMIT
from src.tennis_scene.evaluation.player_pose.scoring import Predictions
from src.utils.inference_benchmark import (
    Backend,
    Batch,
    BatchStages,
    BenchmarkAdapter,
    BenchmarkCase,
    ConditionUnavailable,
    Status,
    benchmark,
)
from src.utils.inference_benchmark_inputs import PinnedBenchmarkInputs
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


def _require_harness_revision(commit: str) -> None:
    if len(commit) != 40 or any(value not in "0123456789abcdef" for value in commit):
        raise ValueError("Specify the complete benchmark harness commit SHA")
    result = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "--no-ext-diff",
            commit,
            "--",
            "src/utils/inference_benchmark.py",
            "src/utils/inference_benchmark_inputs.py",
        ],
        check=False,
    )
    if result.returncode:
        raise ValueError(f"The benchmark harness differs from {commit}")


def one_case(args: argparse.Namespace) -> Status:
    if args.backend not in MEASURED_BACKENDS or args.parity != PARITY_NOT_MEASURED:
        raise ValueError(
            "Select PT16 or TRT16 and explicitly set parity to not_measured"
        )
    _require_harness_revision(args.harness_commit)
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
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
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
            "harness_commit": args.harness_commit,
            "input_decode": "predecoded BGR",
            "cuda_stream": "non-default",
            "host_threads": torch.get_num_threads(),
            "opencv_threads": cv2.getNumThreads(),
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

    report = benchmark(
        case,
        batches,
        {backend: adapter},
        device=device,
        conversion=conversion,
        parity_not_evaluated_reason=PARITY_NOT_MEASURED_REASON,
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
    parser.add_argument("--backend", choices=MEASURED_BACKENDS, required=True)
    parser.add_argument("--parity", choices=(PARITY_NOT_MEASURED,), required=True)
    parser.add_argument("--harness-commit", required=True)
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
    status = one_case(parser.parse_args())
    if status is not Status.OK:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
