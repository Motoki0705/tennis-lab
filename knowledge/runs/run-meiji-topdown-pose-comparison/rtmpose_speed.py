"""Connect one RTMPose PT16 or TRT16 case to the #2496 timing harness."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from pose_contract import PoseInput
from rtmpose import HEIGHT, MODELS, SOURCE_COMMIT, WIDTH, make_adapter
from speed_inputs import InputSet

from src.utils.inference_benchmark import Backend, Batch, BenchmarkCase, benchmark
from src.utils.inference_benchmark_inputs import TimingSchedule
from src.utils.io import write_json_atomic

BACKENDS = (Backend.PT16, Backend.TRT16)
HARNESS_MERGE_COMMIT = "59faa32a28d2f54e6492abdba2429846fa0397e3"
PARITY_NOT_EVALUATED_REASON = "PT32計測はスコープ外（ユーザー判断 2026-11-25）"


def inputs(
    fixed: InputSet, schedule: TimingSchedule
) -> tuple[Batch[tuple[PoseInput, ...]], ...]:
    rows = {row["id"]: row for row in fixed.rows("crops")}
    batches = []
    for ids in schedule.batch_ids:
        samples = tuple(
            PoseInput(fixed.frame(row["frame_id"]), np.asarray(row["xyxy"], np.float32))
            for row in (rows[key] for key in ids)
        )
        batches.append(Batch(ids, samples, processed_items=len(ids)))
    schedule.require_batches([batch.ids for batch in batches])
    return tuple(batches)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    mode, batch_size = Backend(args.backend), args.batch
    fixed = InputSet(args.inputs)
    if fixed.sha256 != args.input_sha256:
        raise ValueError(
            f"Expected the published #2496 manifest SHA-256 at {args.inputs}"
        )
    schedule = fixed.pins.schedule("crops", batch_size)
    batches = inputs(fixed, schedule)
    directory = args.output / args.model / f"{mode}-b{batch_size}"
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    spec = MODELS[args.model]
    case = BenchmarkCase(
        args.model,
        mode,
        batch_size,
        "person-crop",
        fixed.sha256,
        {
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "harness_merge_commit": HARNESS_MERGE_COMMIT,
            "dataset_id": fixed.manifest["id"],
            "mmpose_commit": SOURCE_COMMIT,
            "timing_protocol_sha256": fixed.pins.timing_sha256,
            "timed_input_ids_sha256": schedule.timed_ids_sha256,
            "parameters": spec.parameters,
            "input_hw": [HEIGHT, WIDTH],
            "input_decode": "predecoded BGR",
            "crop_box": "native xyxy from the #2496 crop index",
            "preprocessed_rows": sum(len(batch.ids) for batch in batches),
            "host_threads": torch.get_num_threads(),
            "cuda_stream": "non-default",
            "flip_test": True,
            "forward_calls_per_batch": 2,
            "forward_dtype": "float16",
            "fp32_operations": ["RGB normalization", "flip fusion", "SimCC decoding"],
        },
    )
    adapter = make_adapter(
        model_id=args.model,
        backend=mode,
        checkpoint=args.checkpoint,
        source_root=args.source_root,
        engine=args.engine,
        device=args.device,
    )
    report = benchmark(
        case,
        batches,
        {mode: adapter},
        device=args.device,
        dedicated_stream=True,
        protocol=schedule.protocol,
        parity_not_evaluated_reason=PARITY_NOT_EVALUATED_REASON,
    )
    write_json_atomic(directory / "report.json", report.to_dict())
    print(f"{args.model}/{mode}/B{batch_size} {report.status}")
    if report.failure is not None:
        print(report.failure.message)


if __name__ == "__main__":
    main()
