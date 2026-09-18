"""Compare explicit ViTPose precisions on frozen real-person crops."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.submodules.configuration import ViTPoseHeadConfig
from src.submodules.models import Pose2DRequest, TrackResult, ViTPosePose2D
from src.tennis_scene.dataset_pipeline.quality import summarize
from src.tennis_scene.reference_pipeline.observations import sha256
from src.utils.io import save_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = ViTPosePose2D(
        args.checkpoint.resolve(),
        device="cuda:0",
        flip_test=True,
        batch_size=8,
        head_config=ViTPoseHeadConfig(1280, 17, 2, (256, 256), (4, 4), 1, 0, ()),
    )
    model.load()
    results = {}
    for camera in ("cam0", "cam1"):
        with np.load(args.observations / f"{camera}_people.npz") as saved:
            boxes = saved["boxes"][:, :128]
        tracks = TrackResult({i: torch.from_numpy(b) for i, b in enumerate(boxes)}, 128)
        for person in (0, 1):
            request = Pose2DRequest(
                args.clip / f"media/{camera}.mp4",
                tracks.bbx_xys(person, base_enlarge=1.2),
            )
            timing, predictions = {}, {}
            for precision in ("float32", "bfloat16"):
                model.precision = precision
                model.predict(Pose2DRequest(request.video_path, request.bbx_xys[:8]))
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
    model.unload()
    save_json_atomic(
        {"checkpoint_sha256": sha256(args.checkpoint), "results": results},
        args.output / "metrics.json",
    )
    print(results, flush=True)


if __name__ == "__main__":
    main()
