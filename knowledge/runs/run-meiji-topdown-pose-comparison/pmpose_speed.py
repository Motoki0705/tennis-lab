"""Run one PMPose FP16 condition on the published #2496 timing inputs."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from pmpose import benchmark_settings, make_adapter
from pmpose_assets import WEIGHTS
from pose_contract import PoseInput
from speed_inputs import InputSet

from src.utils.inference_benchmark import Backend, Batch, BenchmarkCase, benchmark
from src.utils.inference_benchmark_inputs import TimingSchedule
from src.utils.io import write_json_atomic

HARNESS_MERGE_COMMIT = "59faa32a28d2f54e6492abdba2429846fa0397e3"
PARITY_NOT_EVALUATED_REASON = "PT32計測はスコープ外（ユーザー判断 2026-11-25）"


def pose_batches(
    fixed: InputSet, schedule: TimingSchedule
) -> list[Batch[tuple[PoseInput, ...]]]:
    crops = {row["id"]: row for row in fixed.rows("crops")}
    batches = []
    for ids in schedule.batch_ids:
        rows = tuple(
            PoseInput(
                fixed.frame(crops[key]["frame_id"]),
                np.asarray(crops[key]["xyxy"], dtype=np.float32),
            )
            for key in ids
        )
        batches.append(Batch(ids, rows, processed_items=len(rows)))
    schedule.require_batches([batch.ids for batch in batches])
    return batches


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=WEIGHTS, required=True)
    parser.add_argument(
        "--backend", type=Backend, choices=(Backend.PT16, Backend.TRT16), required=True
    )
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    fixed = InputSet(args.inputs)
    schedule = fixed.pins.schedule("crops", args.batch)
    directory = args.output / args.model / f"{args.backend}-b{args.batch}"
    directory.mkdir(parents=True, exist_ok=False)
    batches = pose_batches(fixed, schedule)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    execution_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    case = BenchmarkCase(
        args.model,
        args.backend,
        args.batch,
        "person-crop",
        fixed.sha256,
        {
            **benchmark_settings(args.model, args.backend),
            "execution_commit": execution_commit,
            "harness_merge_commit": HARNESS_MERGE_COMMIT,
            "dataset_id": fixed.manifest["id"],
            "timing_protocol_sha256": fixed.pins.timing_sha256,
            "timed_input_ids_sha256": schedule.timed_ids_sha256,
            "input_decode": "predecoded BGR",
            "crop_box": "native xyxy from the #2496 crop index",
            "excluded_tail_ids": fixed.manifest["pose_sweep_tail_ids"],
            "host_threads": torch.get_num_threads(),
            "opencv_threads": cv2.getNumThreads(),
            "cuda_stream": "non-default",
        },
    )
    adapter = make_adapter(
        model_id=args.model,
        backend=args.backend,
        checkpoint=args.assets / WEIGHTS[args.model].filename,
        source_root=args.assets / "BBoxMaskPose",
        engine=args.assets / f"{args.model}-engine"
        if args.backend is Backend.TRT16
        else None,
        device=args.device,
    )
    report = benchmark(
        case,
        batches,
        {args.backend: adapter},
        device=args.device,
        dedicated_stream=True,
        protocol=schedule.protocol,
        parity_not_evaluated_reason=PARITY_NOT_EVALUATED_REASON,
    )
    write_json_atomic(directory / "report.json", report.to_dict())
    print(f"{args.model}/{args.backend}/B{args.batch} {report.status}", flush=True)
    if report.failure is not None:
        print(f"{report.failure.phase}: {report.failure.message}", flush=True)


if __name__ == "__main__":
    main()
