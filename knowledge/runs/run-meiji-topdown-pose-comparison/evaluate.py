"""Run a single explicit pose condition on the fixed #2495 cohort."""

from __future__ import annotations

import argparse
import importlib
import subprocess
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from typing import Any, TypeVar, cast

import cv2
import numpy as np
import torch
from fixed_inputs import RowKey, load_cohort, pose_inputs, row_keys
from numpy.typing import NDArray
from pose_contract import PoseBatch, PoseInput, PoseStages

from src.tennis_scene.evaluation.player_pose.scoring import (
    FrameKey,
    Predictions,
    ReferenceFrame,
    ScoringPolicy,
    score_players,
)
from src.utils.checksum import dual_sha256
from src.utils.device import resolve_device
from src.utils.inference_benchmark import (
    Backend,
    BenchmarkAdapter,
    ConditionUnavailable,
    Stage,
    Status,
)
from src.utils.io import write_json_atomic
from src.utils.schema.player import COCO_KP_NAMES

MODEL_IDS = (
    "vitpose-h",
    "vitpose-h-no-flip",
    "rtmpose-m",
    "rtmpose-x",
    "pmpose-l",
    "pmpose-h",
    "sapiens2-1b",
    "sapiens2-5b",
)
# The scorer requires a finite exclusive cutoff. This accepts all finite
# float32 confidence values without converting any model-specific scores.
POLICY = ScoringPolicy(
    score_cutoffs=(), min_pose_confidence=float(np.finfo(np.float64).min)
)
Value = TypeVar("Value")
Infer = Callable[[tuple[PoseInput, ...], PoseStages], PoseBatch]


class AccuracyStages:
    """Run the same adapter stages with backend checks and no timing."""

    def __init__(self, backend: Backend) -> None:
        self.backend = backend
        self.forward_count = 0

    def stage(
        self, stage: Stage, operation: Callable[[], Value], *, name: str = ""
    ) -> Value:
        if stage is Stage.FORWARD:
            raise ValueError("Declare every forward's backend with forward()")
        return operation()

    def forward(
        self, backend: Backend, operation: Callable[[], Value], *, name: str = ""
    ) -> Value:
        if backend is not self.backend:
            raise ConditionUnavailable(
                Status.BACKEND_MISMATCH,
                f"Requested {self.backend}, forward declared {backend}",
            )
        self.forward_count += 1
        return operation()


def predict_rows(
    inputs: Iterable[tuple[RowKey, PoseInput]],
    infer: Infer,
    backend: Backend,
    batch_size: int,
) -> tuple[tuple[RowKey, ...], PoseBatch]:
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    source = iter(inputs)
    keys: list[RowKey] = []
    keypoints: list[NDArray[np.float32]] = []
    extras: dict[str, list[NDArray[np.generic]]] = defaultdict(list)
    while batch := tuple(islice(source, batch_size)):
        addresses, rows = zip(*batch, strict=True)
        stages = AccuracyStages(backend)
        output = infer(rows, stages)
        if stages.forward_count == 0 or len(output.keypoints_xyc) != len(batch):
            raise ValueError(
                f"Missing forward or changed pose row count at {addresses[0]}"
            )
        if keys and extras.keys() != output.extras.keys():
            raise ValueError(f"Pose extras changed fields at {addresses[0]}")
        for name, values in output.extras.items():
            chunks = extras[name]
            if chunks and (
                values.shape[1:] != chunks[0].shape[1:]
                or values.dtype != chunks[0].dtype
            ):
                raise ValueError(
                    f"Pose extra {name!r} changed shape or dtype at {addresses[0]}"
                )
            chunks.append(values.copy())
        keys.extend(addresses)
        keypoints.append(output.keypoints_xyc.copy())
    if not keys:
        raise ValueError("The evaluation has no eligible top-down boxes")
    return tuple(keys), PoseBatch(
        np.concatenate(keypoints),
        {name: np.concatenate(chunks) for name, chunks in extras.items()},
    )


def score_fixed(
    references: Mapping[FrameKey, ReferenceFrame],
    keys: tuple[RowKey, ...],
    poses: PoseBatch,
) -> dict[str, Any]:
    if keys != row_keys(references) or len(keys) != len(poses.keypoints_xyc):
        raise ValueError(
            "Predicted joint rows do not match the pinned player addresses"
        )
    predictions = {}
    finite_pair_references = {}
    offset = 0
    for key in sorted(references):
        reference = references[key]
        count = len(reference.player_ids)
        # The common IoU scorer must recover the known row association. The
        # identical diagonal boxes give 1.0; a duplicate would make it ambiguous.
        if len(np.unique(reference.boxes_xyxy, axis=0)) != count:
            raise ValueError(
                f"Duplicate reference boxes make row association ambiguous: {key}"
            )
        points = poses.keypoints_xyc[offset : offset + count]
        predictions[key] = Predictions(reference.boxes_xyxy, np.ones(count), points)
        finite_pair_references[key] = ReferenceFrame(
            reference.player_ids,
            reference.boxes_xyxy,
            reference.keypoints_xy,
            reference.joint_valid & np.isfinite(points).all(axis=-1),
        )
        offset += count
    fixed: dict[str, Any] = score_players(references, predictions, policy=POLICY)
    finite: dict[str, Any] = score_players(
        finite_pair_references, predictions, policy=POLICY
    )
    fixed["finite_pairs"] = {
        "overall": finite["overall"]["pose"]["with_detection_coverage"],
        "by_clip": {
            key: value["pose"]["with_detection_coverage"]
            for key, value in finite["by_clip"].items()
        },
        "by_camera": {
            key: value["pose"]["with_detection_coverage"]
            for key, value in finite["by_camera"].items()
        },
        "by_player": {
            clip: {
                key: value["pose"]["with_detection_coverage"]
                for key, value in players.items()
            }
            for clip, players in finite["by_player"].items()
        },
        "by_size": {
            key: value["pose"]["with_detection_coverage"]
            for key, value in finite["by_size"].items()
        },
    }
    return fixed


def save_predictions(
    output: Path,
    references: Mapping[FrameKey, ReferenceFrame],
    keys: tuple[RowKey, ...],
    poses: PoseBatch,
) -> dict[str, str]:
    path = output / "predictions.npz"
    boxes = [references[key].boxes_xyxy for key in sorted(references)]
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            clip_id=np.array([row.frame.clip_id for row in keys]),
            camera_id=np.array([row.frame.camera_id for row in keys]),
            source_frame=np.array(
                [row.frame.source_frame for row in keys], dtype=np.int64
            ),
            player_id=np.array([row.player_id for row in keys], dtype=np.int64),
            bbox_xyxy=np.concatenate(boxes).astype(np.float32),
            keypoints_xyc=poses.keypoints_xyc,
            **{f"extra__{name}": value for name, value in poses.extras.items()},
        )
    return {"path": path.name, "sha256": dual_sha256(path)}


@contextmanager
def accuracy_device(spec: str) -> Iterator[dict[str, Any]]:
    device = resolve_device(torch.device(spec))
    if device.type != "cuda":
        raise ValueError("The pose comparison requires an explicit CUDA device")
    matmul = torch.backends.cuda.matmul
    cudnn = torch.backends.cudnn
    conv = cudnn.conv
    initial = matmul.fp32_precision, cudnn.fp32_precision, conv.fp32_precision
    with (
        torch.cuda.device(device),
        torch.no_grad(),
        torch.autocast("cuda", enabled=False),
    ):
        try:
            matmul.fp32_precision = cudnn.fp32_precision = conv.fp32_precision = "ieee"
            properties = torch.cuda.get_device_properties(device)
            yield {
                "gpu": properties.name,
                "gpu_total_bytes": properties.total_memory,
                "cuda_device": str(device),
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "opencv": cv2.__version__,
                "fp32_precision": {
                    "matmul": matmul.fp32_precision,
                    "cudnn": cudnn.fp32_precision,
                    "cudnn_conv": conv.fp32_precision,
                },
            }
        finally:
            matmul.fp32_precision, cudnn.fp32_precision, conv.fp32_precision = initial


def _assets(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "path": str(path),
            "sha256": dual_sha256(path),
            "bytes": path.stat().st_size,
        }
        for name, path in paths.items()
    }


def _adapter(
    args: argparse.Namespace, backend: Backend
) -> BenchmarkAdapter[tuple[PoseInput, ...], PoseBatch]:
    module = importlib.import_module(args.model.split("-", 1)[0])
    adapter = module.make_adapter(
        model_id=args.model,
        backend=backend,
        checkpoint=args.checkpoint.resolve(),
        device=args.device,
        engine=None if args.engine is None else args.engine.resolve(),
        source_root=None if args.source_root is None else args.source_root.resolve(),
    )
    if adapter.backend is not backend:
        raise ConditionUnavailable(
            Status.BACKEND_MISMATCH,
            f"Requested {backend}, adapter declares {adapter.backend}",
        )
    return cast(BenchmarkAdapter[tuple[PoseInput, ...], PoseBatch], adapter)


def run(args: argparse.Namespace) -> None:
    backend = Backend(args.backend)
    if args.batch_size < 1 or (
        args.engine is not None and backend is not Backend.TRT16
    ):
        raise ValueError("Supply a positive batch size and use engine only with TRT16")
    args.output.mkdir(parents=True, exist_ok=False)
    meta = {
        "schema_version": 1,
        "model": args.model,
        "backend": backend,
        "batch_size": args.batch_size,
        "bundle_sha256": args.bundle_sha256,
        "weights": {},
        "engines": {},
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "joint_order": COCO_KP_NAMES,
    }
    try:
        bundle_id, references, videos = load_cohort(
            args.bundle, args.bundle_sha256, args.media_root
        )
        meta["bundle_id"] = bundle_id
        adapter = _adapter(args, backend)
        if args.checkpoint_sha256 is not None:
            checkpoint = args.checkpoint.resolve()
            if not checkpoint.is_file():
                raise ConditionUnavailable(
                    Status.UNAVAILABLE, f"Missing checkpoint: {checkpoint}"
                )
            checkpoint_sha256 = dual_sha256(checkpoint)
            if checkpoint_sha256 != args.checkpoint_sha256:
                raise ValueError(
                    f"Checkpoint SHA-256 differs from the requested pin: {checkpoint}"
                )
        if all(path.is_file() for path in adapter.weights.values()):
            meta["weights"] = _assets(adapter.weights)
        if all(path.is_file() for path in adapter.engines.values()):
            meta["engines"] = _assets(adapter.engines)
        cv2.setNumThreads(1)
        with accuracy_device(args.device) as environment:
            meta["environment"] = environment
            infer = cast(Infer, adapter.load())
            keys, poses = predict_rows(
                pose_inputs(videos, references), infer, backend, args.batch_size
            )
        if not meta["weights"]:
            meta["weights"] = _assets(adapter.weights)
        if not meta["engines"]:
            meta["engines"] = _assets(adapter.engines)
        scores = score_fixed(references, keys, poses)
        predictions = save_predictions(args.output, references, keys, poses)
        write_json_atomic(args.output / "scores.json", scores)
        write_json_atomic(
            args.output / "receipt.json",
            {
                **meta,
                "status": Status.OK,
                "rows": len(keys),
                "predictions": predictions,
                "scores": {
                    "path": "scores.json",
                    "sha256": dual_sha256(args.output / "scores.json"),
                },
            },
        )
    except Exception as error:
        status = "error"
        if isinstance(error, ConditionUnavailable):
            status = error.status
        elif isinstance(error, torch.OutOfMemoryError):
            status = Status.OOM
        write_json_atomic(
            args.output / "receipt.json",
            {**meta, "status": status, "error": f"{type(error).__name__}: {error}"},
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_IDS, required=True)
    parser.add_argument("--backend", choices=list(Backend), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
