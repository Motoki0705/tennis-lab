"""Compare explicit ViTPose precisions on frozen real-person crops."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from src.submodules.models import Pose2DRequest, TrackResult, ViTPosePose2D
from src.tennis_scene.dataset_pipeline.quality import summarize
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.tennis_scene.reference_pipeline.observations import sha256
from src.utils.io import save_json_atomic

from .configuration import ViTPoseBenchmarkConfig
from .media import sample_frames


def benchmark(args: ViTPoseBenchmarkConfig) -> None:
    if args.output.exists():
        raise FileExistsError(f"Choose a new benchmark run: {args.output}")
    clip = ClipManifest.load(args.clip)
    if args.frames > clip.num_frames:
        raise ValueError("Requested frame count exceeds clip")
    for camera in args.cameras:
        sample_frames(clip, camera, np.asarray([], dtype=int))
    args.output.mkdir(parents=True, exist_ok=False)
    model = ViTPosePose2D(
        args.checkpoint.resolve(),
        device=args.device,
        flip_test=args.flip_test,
        batch_size=args.batch_size,
        head_config=args.head,
    )
    model.load()
    try:
        results = _compare(args, model, clip)
    finally:
        model.unload()
    save_json_atomic(
        {"checkpoint_sha256": sha256(args.checkpoint), "results": results},
        args.output / "metrics.json",
    )
    print(results, flush=True)


def _compare(
    args: ViTPoseBenchmarkConfig, model: ViTPosePose2D, clip: ClipManifest
) -> dict[str, Any]:
    results = {}
    for camera in args.cameras:
        with np.load(args.observations / f"{camera}_people.npz") as saved:
            boxes = saved["boxes"][:, : args.frames]
        if (
            boxes.ndim != 3
            or boxes.shape[1:] != (args.frames, 4)
            or max(args.people) >= len(boxes)
            or not np.isfinite(boxes).all()
        ):
            raise ValueError(
                "Frozen boxes must cover every requested frame and be finite"
            )
        tracks = TrackResult(
            {i: torch.from_numpy(b) for i, b in enumerate(boxes)}, args.frames
        )
        for person in args.people:
            request = Pose2DRequest(
                clip.media_path(camera),
                tracks.bbx_xys(person, base_enlarge=args.crop_enlarge),
            )
            timing, predictions = {}, {}
            for precision in ("float32", "bfloat16"):
                model.precision = precision
                model.predict(
                    Pose2DRequest(
                        request.video_path, request.bbx_xys[: args.warmup_frames]
                    )
                )
                torch.cuda.synchronize()
                start = time.perf_counter()
                predictions[precision] = model.predict(request).keypoints.numpy()
                torch.cuda.synchronize()
                timing[precision] = time.perf_counter() - start
            a, b = predictions["float32"], predictions["bfloat16"]
            key = f"{camera}_person_{person}"
            results[key] = {
                "seconds": timing,
                "coordinate_difference_px": summarize(
                    np.linalg.norm(a[..., :2] - b[..., :2], axis=-1)
                ),
                "confidence_difference": summarize(np.abs(a[..., 2] - b[..., 2])),
            }
            np.savez_compressed(args.output / f"{key}.npz", float32=a, bfloat16=b)
    return results
