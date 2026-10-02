"""CPU proposals and motion-blur refinement; candidates never become labels automatically."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.chat_annotation.runtime.contracts import (
    ClipManifest,
)

from .common import (
    atomic_write_json,
    compact_ranges,
    iter_frames,
    utc_now,
)
from .configuration import paths
from .worker_context import Ctx, dump


def read_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found; run the candidate command first"
        )
    from .configuration import json_object

    return json_object(path)


class BallModel:
    """ConvNeXt MDD ball model, loaded once. Workers run it on CPU; the prefetcher may use CUDA
    (queued through the training queue). The device does not enter the cache identity."""

    def __init__(self, threads: int = 2, device: str = "cpu") -> None:
        import torch

        from src.tasks.ball_detection.inference.checkpoint import load_ball_checkpoint
        from src.tasks.ball_detection.inference.predictor import BallDetectionPredictor

        self.torch = torch
        self.threads = threads
        self.device = torch.device(device)
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        loaded = load_ball_checkpoint(
            paths().checkpoint(), strict=True, weights_only=False
        )
        self.predictor = BallDetectionPredictor(
            loaded.model_io,
            self.device,
            subpixel_refine=True,
            image_normalization=loaded.image_normalization,
        )
        self.window = self.predictor.configured_frames
        self.height, self.width = map(int, loaded.config.data.image_size)

    def identity(
        self, manifest: ClipManifest, threshold: float, max_peaks: int
    ) -> dict[str, Any]:
        return ball_identity(manifest, threshold, max_peaks)


def ball_identity(
    manifest: ClipManifest, threshold: float, max_peaks: int
) -> dict[str, Any]:
    """A checkpoint content hash identifies all model/window/image configuration."""
    paths().checkpoint()
    return {
        "video_sha256": manifest.sha256,
        "checkpoint_sha256": paths().ball_checkpoint_sha256,
        "threshold": threshold,
        "max_peaks": max_peaks,
    }


def compute_ball_candidates(
    model: BallModel,
    video: Path,
    manifest: ClipManifest,
    work_dir: Path | None,
    path: Path,
    start: int,
    stop: int,
    threshold: float,
    max_peaks: int,
    max_seconds: float,
) -> dict[str, Any]:
    from src.tasks.ball_detection.visualization.inference.peaks import (
        decode_frame_peaks,
    )

    n, w, h, window = (
        len(manifest.frames),
        manifest.width,
        manifest.height,
        model.window,
    )
    if n < window:
        raise ValueError("clip shorter than the model window")
    identity = model.identity(manifest, threshold, max_peaks)
    cache: dict[str, Any] = (
        read_cache(path)
        if path.exists()
        else {
            "schema": "campaign_ball_candidates.v1",
            "proposals_only": True,
            "identity": identity,
            "frames": {},
        }
    )
    if cache["identity"] != identity:
        raise ValueError(f"existing {path.name} was made with other settings")
    blocks = [
        (min(b, n - window), max(start, b), min(stop, b + window))
        for b in range((start // window) * window, stop, window)
    ]
    todo = [
        blk
        for blk in blocks
        if any(str(i) not in cache["frames"] for i in range(blk[1], blk[2]))
    ]
    began = time.monotonic()
    done_windows = 0
    # One sequential decode; model inputs are kept for the frames of pending windows only.
    needed = sorted({i for ws, _, _ in todo for i in range(ws, ws + window)})
    inputs: dict[int, NDArray[np.uint8]] = {}
    frames_iter = (
        iter_frames(video, manifest, needed[0], needed[-1] + 1, work_dir)
        if needed
        else iter([])
    )
    for window_start, emit_start, emit_stop in todo:
        if time.monotonic() - began > max_seconds:
            break
        while window_start + window - 1 not in inputs:
            index, bgr = next(frames_iter)
            if index in needed:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                inputs[index] = cv2.resize(
                    rgb, (model.width, model.height), interpolation=cv2.INTER_LINEAR
                )
        batch = [inputs[i] for i in range(window_start, window_start + window)]
        for old in [i for i in inputs if i < window_start]:
            del inputs[old]
        tensor = (
            model.torch.from_numpy(np.stack(batch))
            .permute(0, 3, 1, 2)
            .unsqueeze(0)
            .float()
            / 255
        )
        model.torch.set_num_threads(model.threads)
        with model.torch.no_grad():
            prediction = model.predictor.predict(tensor)
        peaks = decode_frame_peaks(
            prediction.heatmaps[0],
            original_size=(w, h),
            threshold=threshold,
            nms_kernel=5,
            max_peaks=max_peaks,
            subpixel_refine=True,
        )
        for index in range(emit_start, emit_stop):
            found = peaks[index - window_start]
            cache["frames"][str(index)] = {
                "candidates": [
                    {
                        "center_px": [round(float(x), 2), round(float(y), 2)],
                        "score": round(float(s), 4),
                    }
                    for (x, y), s in zip(found.points, found.scores, strict=True)
                    if 0 <= x < w and 0 <= y < h
                ]
            }
        done_windows += 1
        cache["updated_at"] = utc_now()
        atomic_write_json(path, cache, indent=None)
    frames_iter.close() if hasattr(frames_iter, "close") else None
    missing = [i for i in range(start, stop) if str(i) not in cache["frames"]]
    return {
        "cache": str(path),
        "windows_run": done_windows,
        "seconds": round(time.monotonic() - began, 1),
        "cached_frames": len(cache["frames"]),
        "remaining_in_range": compact_ranges(missing)[:20],
        "rerun_to_continue": bool(missing),
    }


def cmd_cands_ball(ctx: Ctx, args: argparse.Namespace) -> int:
    """Ball-model proposals (search guides, never labels): prefetched copy if available."""
    start, stop = ctx.check_range(args.start, args.stop)
    path = ctx.work / "cands_ball.json"
    prefetched = (
        paths().campaign_dir / "cache" / "cands_ball"
    ) / f"{ctx.task['clip_id']}.json"
    identity = ball_identity(ctx.manifest, args.threshold, args.max_peaks)
    if prefetched.exists():
        shared = read_cache(prefetched)
        if shared.get("identity") != identity:
            raise ValueError(
                "shared candidate cache belongs to a different checkpoint or configuration"
            )
        if shared.get("identity") == identity and all(
            str(i) in shared["frames"] for i in range(start, stop)
        ):
            local = read_cache(path) if path.exists() else None
            if local is not None and local.get("identity") != identity:
                raise ValueError(
                    "existing cands_ball.json was made with other settings"
                )
            merged = (
                shared
                if local is None
                else {**local, "frames": {**shared["frames"], **local["frames"]}}
            )
            atomic_write_json(path, merged, indent=None)
            dump(
                {
                    "cache": str(path),
                    "source": "prefetched (shared cache; no model run)",
                    "cached_frames": len(merged["frames"]),
                    "refined": any(
                        "blob" in c
                        for f in merged["frames"].values()
                        for c in f["candidates"]
                    ),
                    "rerun_to_continue": False,
                }
            )
            return 0
    slot = acquire_model_slot(
        lambda: prefetched_ready(prefetched, identity, start, stop)
    )
    if slot == "prefetched":
        return cmd_cands_ball(
            ctx, args
        )  # the shared cache appeared while waiting: copy it
    if slot is None:
        dump(
            {
                "source": "busy",
                "rerun_to_continue": True,
                "note": f"all {LOCAL_MODEL_SLOTS} local model slots are in use (host memory guard). Do the visual "
                "overview first and rerun cands-ball later; the shared cache may be ready by then.",
            }
        )
        return 0
    try:
        summary = compute_ball_candidates(
            BallModel(threads=2),
            ctx.video,
            ctx.manifest,
            ctx.work,
            path,
            start,
            stop,
            args.threshold,
            args.max_peaks,
            args.max_seconds,
        )
    finally:
        slot.close()
    dump({"source": "computed locally", **summary})
    return 0


LOCAL_MODEL_SLOTS = (
    4  # concurrent CPU ball-model runs across all workers (~1.3 GB each)
)


def prefetched_ready(
    path: Path, identity: dict[str, Any], start: int, stop: int
) -> bool:
    if not path.exists():
        return False
    shared = read_cache(path)
    return shared.get("identity") == identity and all(
        str(i) in shared["frames"] for i in range(start, stop)
    )


def acquire_model_slot(recheck: Any, max_wait: float = 240.0) -> Any:
    """One of LOCAL_MODEL_SLOTS flock slots (pre-created files, opened read-only: the worker sandbox
    cannot create files here). Returns the open slot, "prefetched" if the shared cache became
    usable while waiting, or None after max_wait seconds."""
    import fcntl

    deadline = time.monotonic() + max_wait
    while True:
        for i in range(LOCAL_MODEL_SLOTS):
            handle = (paths().locks / f"model_slot_{i}").open("rb")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return handle
            except BlockingIOError:
                handle.close()
        if recheck():
            return "prefetched"
        if time.monotonic() > deadline:
            return None
        time.sleep(5)


def blob_center(
    frames: dict[int, NDArray[np.uint8]], t: int, cx: float, cy: float, radius: int = 40
) -> dict[str, Any] | None:
    """Centroid of the moving bright blob under a candidate: frame - median(t±2..4), 8-connected.

    Gives the middle of a motion-blur streak (model peaks sit near its leading end).
    Returns None when contrast, size or distance checks fail; the worker then decides by eye.
    """
    image = frames[t]
    H, W = image.shape[:2]
    x0, y0 = max(0, int(cx) - radius), max(0, int(cy) - radius)
    x1, y1 = min(W, int(cx) + radius + 1), min(H, int(cy) + radius + 1)
    neighbours = [
        frames[t + d][y0:y1, x0:x1] for d in (-4, -3, -2, 2, 3, 4) if t + d in frames
    ]
    if len(neighbours) < 3:
        return None
    background = np.median(np.stack(neighbours).astype(np.int16), axis=0)
    # Brighter-than-background only: the ball is lighter than every court surface, while its
    # shadow and most occluders are darker, so they do not pull the centroid.
    diff = np.clip(image[y0:y1, x0:x1].astype(np.int16) - background, 0, None).max(
        axis=2
    )
    u, v = cx - x0, cy - y0
    yy, xx = np.mgrid[0 : diff.shape[0], 0 : diff.shape[1]]
    near = (xx - u) ** 2 + (yy - v) ** 2 <= 8**2
    if not near.any():
        return None
    peak = float(diff[near].max())
    if peak < 18:
        return None
    mask = (diff >= max(12.0, 0.25 * peak)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    seed = np.argmax(np.where(near, diff, -1))
    label = labels.flat[seed]
    if label == 0:
        return None
    area = int(stats[label, cv2.CC_STAT_AREA])
    if not 3 <= area <= 2500:
        return None
    ys, xs = np.nonzero(labels == label)
    weights = diff[ys, xs].astype(np.float64)
    points = np.stack([xs, ys], axis=1).astype(np.float64)
    mean = points.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov((points - mean).T))
    axis, perp = vectors[:, -1], vectors[:, 0]
    along = (points - mean) @ axis
    limits = np.asarray(np.percentile(along, [2, 98]), dtype=np.float64)
    low, high = float(limits[0]), float(limits[1])
    across = float(((points - mean) @ perp * weights).sum() / weights.sum())
    # Streak middle along its main axis (geometric, not brightness-weighted); weighted across it.
    middle = mean + axis * (low + high) / 2 + perp * across
    length = float(high - low)
    center = [round(float(middle[0]) + x0, 1), round(float(middle[1]) + y0, 1)]
    if math.dist(center, (cx, cy)) > 30:
        return None
    return {"center_px": center, "area": area, "length": round(length, 1)}


def refine_candidates(
    video: Path,
    manifest: ClipManifest,
    work_dir: Path | None,
    path: Path,
    start: int,
    stop: int,
    top: int,
) -> dict[str, Any]:
    """Add blob (streak-middle) centers to cached ball candidates that do not have one yet."""
    cache = read_cache(path)
    n = len(manifest.frames)
    targets = sorted(
        int(k)
        for k, v in cache["frames"].items()
        if start <= int(k) < stop
        and any("blob" not in c for c in v["candidates"][:top])
    )
    if not targets:
        return {
            "cache": str(path),
            "refined": 0,
            "note": "all candidates in range already have blob centers (or none exist)",
        }
    lo, hi = max(0, targets[0] - 4), min(n, targets[-1] + 5)
    buffer: dict[int, NDArray[np.uint8]] = {}
    refined = failed = 0
    pending = list(targets)
    for index, image in iter_frames(video, manifest, lo, hi, work_dir):
        buffer[index] = image
        while pending and (pending[0] + 4 <= index or index == hi - 1):
            t = pending.pop(0)
            for cand in cache["frames"][str(t)]["candidates"][:top]:
                if "blob" in cand:
                    continue
                blob = blob_center(buffer, t, *cand["center_px"])
                cand["blob"] = blob
                refined += blob is not None
                failed += blob is None
            for old in [
                k for k in buffer if k < (pending[0] - 4 if pending else index)
            ]:
                del buffer[old]
    cache["updated_at"] = utc_now()
    atomic_write_json(path, cache, indent=None)
    return {
        "cache": str(path),
        "candidates_refined": refined,
        "candidates_without_blob": failed,
        "note": "blob.center_px = middle of the moving streak; verify visually (crops --source cands-ball shows both)",
    }


def cmd_refine_ball(ctx: Ctx, args: argparse.Namespace) -> int:
    start, stop = ctx.check_range(args.start, args.stop)
    dump(
        refine_candidates(
            ctx.video,
            ctx.manifest,
            ctx.work,
            ctx.work / "cands_ball.json",
            start,
            stop,
            args.top,
        )
    )
    return 0
