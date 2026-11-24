"""Connect one RTMPose case to the #2496 inputs and timing harness."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import torch
from parity import compare
from pose_contract import PoseBatch, PoseInput
from rtmpose import HEIGHT, MODELS, SOURCE_COMMIT, WIDTH, FloatArray, make_adapter
from speed_inputs import InputSet

from src.utils.checksum import dual_sha256
from src.utils.inference_benchmark import (
    Backend,
    Batch,
    BenchmarkCase,
    BenchmarkProtocol,
    ConditionUnavailable,
    ParityResult,
    Status,
    benchmark,
)
from src.utils.io import write_json_atomic

COMMON_BATCHES = (1, 4, 8)
PROFILE_MAX_BATCH = 16


def inputs(
    fixed: InputSet, batch_size: int
) -> tuple[Batch[tuple[PoseInput, ...]], ...]:
    count = fixed.manifest["pose_sweep_crop_count"]
    rows = fixed.rows("crops")[:count]
    if not rows or len(rows) != count or count % batch_size:
        raise ValueError(
            f"The #2496 crop cohort cannot fill batch {batch_size}: {len(rows)} rows"
        )
    batches = []
    for start in range(0, count, batch_size):
        chunk = rows[start : start + batch_size]
        ids = tuple(row["id"] for row in chunk)
        samples = tuple(
            PoseInput(fixed.frame(row["frame_id"]), np.asarray(row["xyxy"], np.float32))
            for row in chunk
        )
        batches.append(
            Batch(ids, samples, items=len(chunk), processed_items=len(chunk))
        )
    return tuple(batches)


def save_predictions(
    path: Path, predictions: Iterator[tuple[Batch[tuple[PoseInput, ...]], PoseBatch]]
) -> dict[str, FloatArray]:
    poses = {}
    for batch, output in predictions:
        for key, pose in zip(batch.ids, output.keypoints_xyc, strict=True):
            if key in poses:
                raise ValueError(f"Duplicate speed prediction: {key}")
            poses[key] = pose
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            row_ids=np.array(list(poses)),
            keypoints_xyc=np.stack(list(poses.values())),
        )
    return poses


def load_reference(directory: Path, manifest_sha256: str) -> dict[str, FloatArray]:
    report = directory / "report.json"
    if report.is_file():
        metadata = json.loads(report.read_bytes())
        if (
            metadata["status"] == "ok"
            and metadata["case"]["input_manifest_sha256"] == manifest_sha256
        ):
            path = directory / "predictions.npz"
            if not path.is_file() or metadata.get("predictions_sha256") != dual_sha256(
                path
            ):
                raise ConditionUnavailable(
                    Status.PARITY_UNAVAILABLE,
                    f"PT32 reference predictions missing or changed: {path}",
                )
            with np.load(path, allow_pickle=False) as saved:
                ids = saved["row_ids"].tolist()
                poses = saved["keypoints_xyc"]
            if len(set(ids)) != len(ids):
                raise ValueError(f"Duplicate PT32 reference crop ID: {directory}")
            return dict(zip(ids, poses, strict=True))
    raise ConditionUnavailable(
        Status.PARITY_UNAVAILABLE,
        f"PT32 batch 1 reference missing, failed or measured on different inputs: {directory}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--backend", choices=list(Backend), required=True)
    parser.add_argument(
        "--batch", type=int, choices=(*COMMON_BATCHES, PROFILE_MAX_BATCH), required=True
    )
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
    batches = inputs(fixed, batch_size)
    root = args.output / args.model
    directory = root / f"{mode}-b{batch_size}"
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
            "dataset_id": fixed.manifest["id"],
            "mmpose_commit": SOURCE_COMMIT,
            "parameters": spec.parameters,
            "input_hw": [HEIGHT, WIDTH],
            "input_decode": "predecoded BGR",
            "crop_box": "native xyxy from the #2496 crop index",
            "preprocessed_rows": fixed.manifest["pose_sweep_crop_count"],
            "excluded_tail_ids": fixed.manifest["pose_sweep_tail_ids"],
            "host_threads": torch.get_num_threads(),
            "cuda_stream": "non-default",
            "flip_test": True,
            "forward_calls_per_batch": 2,
            "forward_dtype": "float32" if mode is Backend.PT32 else "float16",
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
    predictions_path = directory / "predictions.npz"

    def parity(
        predictions: Iterator[tuple[Batch[tuple[PoseInput, ...]], PoseBatch]],
    ) -> ParityResult:
        actual = save_predictions(predictions_path, predictions)
        reference = (
            actual
            if mode is Backend.PT32 and batch_size == 1
            else load_reference(root / "PT32-b1", fixed.sha256)
        )
        # Use the shared standalone pose parity thresholds for native pixel
        # errors and same-model differences in raw confidence.
        return compare("vitpose-flip", reference, actual)

    sizes = (
        (*COMMON_BATCHES, PROFILE_MAX_BATCH)
        if batch_size == PROFILE_MAX_BATCH
        else COMMON_BATCHES
    )
    with torch.cuda.stream(torch.cuda.Stream(device=args.device)):
        report = benchmark(
            case,
            batches,
            {mode: adapter},
            device=args.device,
            protocol=BenchmarkProtocol(batch_sizes=sizes),
            check_parity=parity,
        )
    payload = report.to_dict()
    if predictions_path.is_file():
        payload["predictions_sha256"] = dual_sha256(predictions_path)
    write_json_atomic(directory / "report.json", payload)
    print(f"{args.model}/{mode}/B{batch_size} {report.status}")
    if report.failure is not None:
        print(report.failure.message)


if __name__ == "__main__":
    main()
