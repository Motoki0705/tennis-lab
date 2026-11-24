"""Run the pinned EdgeCrafter models against the fixed Meiji player benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, TypeVar, cast

import numpy as np
import torch
from numpy.typing import NDArray

from src.submodules.models.ecpose.inference import (
    decode_predictions,
    load_network,
    preprocess_images,
    require_source,
    require_weight,
    validate_decode_policy,
)
from src.submodules.models.ecpose.releases import (
    IMAGE_SIZE,
    RELEASES,
    SOURCE_COMMIT,
    SOURCE_URL,
)
from src.tennis_scene.evaluation.player_pose.bundle import (
    load_manifest,
    load_reference_frames,
    pinned_file,
)
from src.tennis_scene.evaluation.player_pose.scoring import (
    FrameKey,
    Predictions,
    score_players,
)
from src.utils.checksum import dual_sha256
from src.utils.inference_benchmark import Backend, BatchStages, BenchmarkAdapter, Stage
from src.utils.io import write_json_atomic
from src.utils.models.tensorrt_artifacts import PLAN_FILE, EngineBundle
from src.utils.models.tensorrt_runtime import TensorRTRunner
from src.utils.video.reader import OpenCVVideoFrameReader
from src.utils.video.types import FramePacket
from src.utils.video.windows import chunked

PREDICTION_SCHEMA = "player_model_predictions_v1"
HEADS = ("logits", "coordinates")
INPUT_NAME = "pixels"
ASSETS = Path("outputs/player-model-comparison/assets")
SOURCE = Path("data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v1")
COHORT = SOURCE.with_name("multiview-player-evaluation-v1")
BACKENDS = {backend.value.lower(): backend for backend in Backend}
SCORE_THRESHOLD = 0.0  # Keep all positive-scoring person rows from the official top-k.

Frames = tuple[NDArray[np.uint8], ...]
Forward = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
Value = TypeVar("Value")


def fetch_assets(directory: Path, models: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    repository = directory / "EdgeCrafter"
    if not repository.exists():
        with TemporaryDirectory(prefix=".edgecrafter-", dir=directory) as staging:
            checkout = Path(staging) / "EdgeCrafter"
            subprocess.run(["git", "init", "-q", str(checkout)], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(checkout),
                    "fetch",
                    "--depth",
                    "1",
                    SOURCE_URL,
                    SOURCE_COMMIT,
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(checkout), "checkout", "--detach", "FETCH_HEAD"],
                check=True,
            )
            require_source(checkout)
            checkout.rename(repository)
    require_source(repository)

    def fetch(model_id: str) -> None:
        release = RELEASES[model_id]
        target = directory / release.filename
        if not target.exists():
            partial_path = target.with_suffix(".pth.part")
            if not partial_path.exists() or partial_path.stat().st_size < release.size:
                subprocess.run(
                    [
                        "curl",
                        "--fail",
                        "--location",
                        "--silent",
                        "--show-error",
                        "--retry",
                        "2",
                        "--connect-timeout",
                        "15",
                        "--speed-limit",
                        "50000",
                        "--speed-time",
                        "40",
                        "--continue-at",
                        "-",
                        "--output",
                        str(partial_path),
                        release.download_url,
                    ],
                    check=True,
                )
            require_weight(partial_path, release)
            partial_path.rename(target)
        else:
            require_weight(target, release)
        print(
            f"{model_id} {target} {release.size} bytes sha256={release.sha256}",
            flush=True,
        )

    models = list(dict.fromkeys(models))
    with ThreadPoolExecutor(max_workers=min(len(models), 3)) as pool:
        list(pool.map(fetch, models))


def _load_forward(
    model_id: str,
    assets: Path,
    backend: Backend,
    device: torch.device,
    engine: Path | None,
) -> Forward:
    release = RELEASES[model_id]
    weights = assets / release.filename
    if backend is Backend.TRT16:
        if engine is None:
            raise ValueError("TRT16 requires an explicit engine directory")
        require_weight(weights, release)
        bundle = EngineBundle.read(engine)
        if (
            bundle.export.model != model_id
            or bundle.export.sources != {"weights": release.sha256}
            or bundle.export.settings.get("source_commit") != SOURCE_COMMIT
            or bundle.export.inputs.keys() != {INPUT_NAME}
            or bundle.export.outputs.keys() != set(HEADS)
        ):
            raise ValueError(
                f"TensorRT engine differs from the pinned EdgeCrafter model: {engine}"
            )
        runner = TensorRTRunner(bundle, device)

        def forward_trt(pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            outputs = runner({INPUT_NAME: pixels})
            return outputs[HEADS[0]], outputs[HEADS[1]]

        return forward_trt
    if engine is not None:
        raise ValueError(f"{backend} accepts no engine directory")
    if backend is Backend.PT16 and device.type != "cuda":
        raise ValueError("PT16 requires CUDA")
    network = load_network(assets / "EdgeCrafter", weights, release).to(device)

    def forward_pt(pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(
            device.type, dtype=torch.float16, enabled=backend is Backend.PT16
        ):
            logits, coordinates = network(pixels)
        return logits, coordinates

    return forward_pt


def _infer(
    frames: Frames,
    stages: BatchStages | None,
    *,
    forward: Forward,
    model_id: str,
    backend: Backend,
    device: torch.device,
    pose_box_margin: float | None,
) -> list[Predictions]:
    def stage(kind: Stage, operation: Callable[[], Value]) -> Value:
        return operation() if stages is None else stages.stage(kind, operation)

    pixels, sizes = stage(Stage.PREPROCESS, lambda: preprocess_images(frames))
    device_pixels, device_sizes = stage(
        Stage.H2D, lambda: (pixels.to(device), sizes.to(device))
    )
    raw = (
        forward(device_pixels)
        if stages is None
        else stages.measure_forward(backend, lambda: forward(device_pixels), name=model_id)
    )
    decoded = stage(
        Stage.POSTPROCESS,
        lambda: decode_predictions(
            RELEASES[model_id],
            raw,
            device_sizes,
            confidence=SCORE_THRESHOLD,
            pose_box_margin=pose_box_margin,
        ),
    )

    def copy_to_host() -> list[Predictions]:
        return [
            Predictions(
                row.boxes_xyxy.cpu().numpy(),
                row.scores.cpu().numpy(),
                None if row.keypoints_xyc is None else row.keypoints_xyc.cpu().numpy(),
            )
            for row in decoded
        ]

    return stage(Stage.D2H, copy_to_host)


def make_benchmark_adapter(
    model_id: str,
    assets: Path,
    backend: Backend,
    device: torch.device,
    *,
    pose_box_margin: float | None = None,
    engine: Path | None = None,
) -> BenchmarkAdapter[Frames, list[Predictions]]:
    """Use the #2496 stage measurement API on already decoded native BGR images."""
    release = RELEASES[model_id]
    validate_decode_policy(release, SCORE_THRESHOLD, pose_box_margin)

    def load() -> Callable[[Frames, BatchStages], list[Predictions]]:
        return partial(
            _infer,
            forward=_load_forward(model_id, assets, backend, device, engine),
            model_id=model_id,
            backend=backend,
            device=device,
            pose_box_margin=pose_box_margin,
        )

    engines = {} if engine is None else {"engine": engine / PLAN_FILE}
    return BenchmarkAdapter(
        backend, load, {"weights": assets / release.filename}, engines
    )


def _evaluation_batches(
    manifest: dict[str, Any],
    source: Path,
    batch_size: int,
) -> Generator[tuple[list[FrameKey], Frames], None, None]:
    for clip in manifest["clips"]:
        indices = clip["evaluation_frames"]
        frame_set = set(indices)
        cameras = clip["source_videos"]
        if [camera["camera_id"] for camera in cameras] != clip["camera_ids"]:
            raise ValueError(
                f"Source-video order differs from the cohort for {clip['clip_id']}"
            )
        for camera in cameras:
            media = pinned_file(source, camera)
            stream = cast(
                Generator[FramePacket[NDArray[np.uint8]], None, None],
                iter(OpenCVVideoFrameReader(media, max_frames=indices[-1] + 1)),
            )
            with closing(stream):
                selected = (packet for packet in stream if packet.index in frame_set)
                for group in chunked(indices, batch_size):
                    frames, keys = [], []
                    for index in group:
                        packet = next(selected, None)
                        if (
                            packet is None
                            or packet.index != index
                            or packet.original_size != (clip["width"], clip["height"])
                        ):
                            raise ValueError(
                                f"Missing or invalid frame {index} in {media}"
                            )
                        frames.append(packet.frame)
                        keys.append(
                            FrameKey(clip["clip_id"], camera["camera_id"], index)
                        )
                    yield keys, tuple(frames)


def _cohort(args: argparse.Namespace) -> dict[str, Any]:
    manifest: dict[str, Any] = load_manifest(
        args.cohort, expected_sha256=args.manifest_sha256
    )
    if (
        dual_sha256(args.source / "manifest.json")
        != manifest["source_benchmark"]["manifest_sha256"]
    ):
        raise ValueError(
            "The original video benchmark differs from the cohort source pin"
        )
    return manifest


def _record(key: FrameKey, prediction: Predictions) -> dict[str, object]:
    return {
        **key._asdict(),
        "boxes_xyxy": prediction.boxes_xyxy.tolist(),
        "scores": prediction.scores.tolist(),
        "keypoints_xyc": None
        if prediction.keypoints_xyc is None
        else prediction.keypoints_xyc.tolist(),
    }


def evaluate(args: argparse.Namespace, *, smoke: bool = False) -> None:
    manifest = _cohort(args)
    release = RELEASES[args.model]
    validate_decode_policy(release, SCORE_THRESHOLD, args.pose_box_margin)
    backend = BACKENDS[args.backend]
    device = torch.device(args.device)
    if device.type != "cuda" and not smoke:
        raise ValueError("Full evaluation requires CUDA; use smoke for CPU validation")
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    forward = _load_forward(args.model, args.assets, backend, device, args.engine)
    batches = _evaluation_batches(manifest, args.source, args.batch_size)
    infer = partial(
        _infer,
        stages=None,
        forward=forward,
        model_id=args.model,
        backend=backend,
        device=device,
        pose_box_margin=args.pose_box_margin,
    )
    if smoke:
        try:
            keys, frames = next(batches)
            rows = [
                _record(key, prediction)
                for key, prediction in zip(keys, infer(frames), strict=True)
            ]
            write_json_atomic(
                args.output,
                {
                    "model_id": args.model,
                    "backend": args.backend,
                    "device": str(device),
                    "benchmark_id": manifest["benchmark_id"],
                    "manifest_sha256": args.manifest_sha256,
                    "weight_sha256": release.sha256,
                    "source_commit": SOURCE_COMMIT,
                    "image_batch_size": args.batch_size,
                    "pose_bbox_joint_margin": args.pose_box_margin,
                    "rows": rows,
                },
            )
            print(
                f"{args.model} {args.backend} {device} forwarded {len(rows)} fixed images to {args.output}"
            )
        finally:
            batches.close()
        return

    references = load_reference_frames(
        args.cohort, expected_sha256=args.manifest_sha256
    )
    header = {
        "schema": PREDICTION_SCHEMA,
        "benchmark_id": manifest["benchmark_id"],
        "manifest_sha256": args.manifest_sha256,
        "model_id": args.model,
        "backend": args.backend,
        "has_pose": release.has_pose,
    }
    predictions = {}
    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "predictions.jsonl"
    with NamedTemporaryFile(
        mode="w", dir=args.output, encoding="utf-8", suffix=".partial", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(json.dumps(header, allow_nan=False) + "\n")
            for keys, frames in batches:
                for key, prediction in zip(keys, infer(frames), strict=True):
                    if key in predictions or key not in references:
                        raise ValueError(
                            f"Unexpected or duplicate evaluation image {key}"
                        )
                    predictions[key] = prediction
                    handle.write(
                        json.dumps(
                            _record(key, prediction),
                            allow_nan=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
            if predictions.keys() != references.keys():
                raise ValueError(
                    f"Missing {len(references.keys() - predictions.keys())} evaluation images"
                )
            metrics = score_players(references, predictions)
            handle.flush()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        finally:
            batches.close()
    temporary.replace(raw_path)
    write_json_atomic(
        args.output / "metrics.json",
        {
            **header,
            "predictions_sha256": dual_sha256(raw_path),
            "weight_sha256": release.sha256,
            "source_commit": SOURCE_COMMIT,
            "input_size_hw": IMAGE_SIZE,
            "image_batch_size": args.batch_size,
            "person_confidence_exclusive": SCORE_THRESHOLD,
            "pose_bbox_joint_margin": args.pose_box_margin,
            "engine_sha256": None
            if args.engine is None
            else dual_sha256(args.engine / PLAN_FILE),
            "score": metrics,
        },
    )
    print(
        f"{args.model} {args.backend} scored {len(predictions)} fixed images in {args.output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    assets = commands.add_parser(
        "assets", help="Fetch and verify the pinned official releases"
    )
    assets.add_argument("--assets", type=Path, default=ASSETS)
    assets.add_argument("--model", choices=RELEASES, action="append")
    for name in ("smoke", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--model", choices=RELEASES, required=True)
        command.add_argument("--backend", choices=BACKENDS, required=True)
        command.add_argument("--assets", type=Path, default=ASSETS)
        command.add_argument("--source", type=Path, default=SOURCE)
        command.add_argument("--cohort", type=Path, default=COHORT)
        command.add_argument("--manifest-sha256", required=True)
        command.add_argument("--batch-size", default=1, type=int)
        command.add_argument("--pose-box-margin", type=float)
        command.add_argument("--engine", type=Path)
        command.add_argument("--device", default="cpu" if name == "smoke" else "cuda:0")
        command.add_argument(
            "--output",
            type=Path,
            required=True,
            help="JSON file for smoke or result directory for evaluation",
        )
    args = parser.parse_args()
    if args.command == "assets":
        fetch_assets(args.assets, args.model or list(RELEASES))
    else:
        with torch.no_grad():
            evaluate(args, smoke=args.command == "smoke")


if __name__ == "__main__":
    main()
