"""Cached-batch producer for MAE training.

This module generates fully preprocessed batches (resize/augment/normalize) in a
background-friendly way, so the training DataLoader can remain lightweight.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor

from src.developing.mae.data.cache.manifest import BatchEntry, EpochCacheManifest
from src.developing.mae.data.cache.paths import EpochCachePaths
from src.developing.mae.data.catalog import VideoCatalog
from src.developing.mae.data.planning import EpochPlan, plan_epoch

try:
    import cv2

    HAS_CV2 = True
except ImportError:  # pragma: no cover
    HAS_CV2 = False

try:
    from decord import VideoReader, cpu

    HAS_DECORD = True
except ImportError:  # pragma: no cover
    HAS_DECORD = False


@dataclass(frozen=True)
class PreprocessConfig:
    patch_size: int = 16
    scale_min: float = 0.8
    scale_max: float = 1.2
    hflip_prob: float = 0.5
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    output_dtype: str = "float16"


@dataclass(frozen=True)
class CacheProducerConfig:
    cache_root: str = "data/mae/cache"
    video_dir: str = "data/tennis/raw/videos"
    use_decord: bool = True
    min_frames: int = 10
    samples_per_video: int = 4
    buckets: tuple[int, ...] = (256, 320, 384, 448, 512, 640, 768, 1024)
    bucket_alpha: float = 2.0
    base_batch_size: int = 32
    min_batch_size: int = 1
    upsample_limit: float = 1.0
    frame_sample_ratio: float = 1.0
    seed: int = 42
    val_split: float = 0.1
    static_val: bool = True
    preprocess: PreprocessConfig = PreprocessConfig()


class VideoFrameReader:
    def __init__(self, *, use_decord: bool) -> None:
        self.use_decord = bool(use_decord and HAS_DECORD)
        if not self.use_decord and not HAS_CV2:
            raise RuntimeError("Neither decord nor cv2 is available for video decoding.")

    def read_many(self, video_path: str, frame_indices: Sequence[int]) -> list[Tensor]:
        if not frame_indices:
            return []
        indices = [int(i) for i in frame_indices]
        if self.use_decord:
            vr = VideoReader(video_path, ctx=cpu(0))
            batch = vr.get_batch(indices).asnumpy()
            frames = [torch.from_numpy(f).permute(2, 0, 1).float() / 255.0 for f in batch]
            return frames

        frames: list[Tensor] = []
        cap = cv2.VideoCapture(video_path)
        try:
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    frames.append(torch.rand(3, 32, 32))
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0)
        finally:
            cap.release()
        return frames


def _round_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return max(multiple, (value // multiple) * multiple)


def preprocess_frame(frame: Tensor, *, resolution: int, seed: int, cfg: PreprocessConfig) -> Tensor:
    rng = random.Random(int(seed))

    if rng.random() < float(cfg.hflip_prob):
        frame = torch.flip(frame, dims=[-1])

    c, h, w = frame.shape
    short = min(h, w)
    target_short = int(round(resolution * rng.uniform(float(cfg.scale_min), float(cfg.scale_max))))
    target_short = max(cfg.patch_size, _round_to_multiple(target_short, cfg.patch_size))
    if short <= 0:
        resized = torch.rand(3, resolution, resolution)
    else:
        scale = float(target_short) / float(short)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        resized = torch.nn.functional.interpolate(
            frame.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    _, rh, rw = resized.shape
    if rh < resolution or rw < resolution:
        padded = torch.zeros(3, max(rh, resolution), max(rw, resolution), dtype=resized.dtype)
        padded[:, :rh, :rw] = resized
        resized = padded
        _, rh, rw = resized.shape

    top = rng.randint(0, rh - resolution) if rh > resolution else 0
    left = rng.randint(0, rw - resolution) if rw > resolution else 0
    cropped = resized[:, top : top + resolution, left : left + resolution]

    mean = torch.tensor(cfg.mean, dtype=cropped.dtype).view(3, 1, 1)
    std = torch.tensor(cfg.std, dtype=cropped.dtype).view(3, 1, 1)
    cropped = (cropped - mean) / std

    if cfg.output_dtype == "float16":
        return cropped.to(dtype=torch.float16)
    if cfg.output_dtype == "float32":
        return cropped.to(dtype=torch.float32)
    raise ValueError(f"Unsupported output_dtype: {cfg.output_dtype}")


def _atomic_replace_dir(tmp_dir: Path, final_dir: Path) -> None:
    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.replace(final_dir)


def _write_done(path: Path) -> None:
    path.write_text("ok\n", encoding="utf-8")


def produce_epoch_cache(
    *,
    cfg: CacheProducerConfig,
    split: str,
    epoch: int,
    plan: EpochPlan,
    catalog: VideoCatalog,
    cache_paths: EpochCachePaths,
) -> Path:
    tmp_dir = cache_paths.epoch_tmp_dir(epoch) if split != "val" else cache_paths.val_dir().with_suffix(".tmp")
    final_dir = cache_paths.epoch_dir(epoch) if split != "val" else cache_paths.val_dir()

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    (tmp_dir / "plan.json").write_text(json.dumps(plan.to_json(), indent=2), encoding="utf-8")

    reader = VideoFrameReader(use_decord=cfg.use_decord)
    video_by_id = {v.video_id: v for v in catalog.videos}

    manifest_entries: list[BatchEntry] = []
    for batch in plan.batches:
        by_video: dict[int, list[int]] = {}
        for s in batch.samples:
            by_video.setdefault(int(s.video_id), []).append(int(s.frame_idx))

        frames_by_key: dict[tuple[int, int], Tensor] = {}
        for vid, indices in by_video.items():
            meta = video_by_id.get(int(vid))
            if meta is None:
                for idx in indices:
                    frames_by_key[(vid, idx)] = torch.rand(3, 32, 32)
                continue
            frames = reader.read_many(meta.path, indices)
            for idx, frame in zip(indices, frames, strict=False):
                frames_by_key[(vid, int(idx))] = frame

        processed: list[Tensor] = []
        video_ids: list[int] = []
        frame_idxs: list[int] = []
        seeds: list[int] = []
        for s in batch.samples:
            frame = frames_by_key.get((int(s.video_id), int(s.frame_idx)))
            if frame is None:
                frame = torch.rand(3, 32, 32)
            img = preprocess_frame(frame, resolution=int(batch.resolution), seed=int(s.seed), cfg=cfg.preprocess)
            processed.append(img)
            video_ids.append(int(s.video_id))
            frame_idxs.append(int(s.frame_idx))
            seeds.append(int(s.seed))

        images = torch.stack(processed, dim=0)
        batch_out = {
            "image": images,
            "resolution": int(batch.resolution),
            "video_ids": torch.tensor(video_ids, dtype=torch.int32),
            "frame_idxs": torch.tensor(frame_idxs, dtype=torch.int32),
            "seeds": torch.tensor(seeds, dtype=torch.int64),
        }

        batch_path = tmp_dir / f"batch_{batch.batch_id:06d}.pt"
        torch.save(batch_out, batch_path)
        manifest_entries.append(
            BatchEntry(
                batch_id=int(batch.batch_id),
                resolution=int(batch.resolution),
                batch_size=int(images.shape[0]),
                path=batch_path.name,
            )
        )

    manifest = EpochCacheManifest(epoch=int(epoch), split=str(split), batches=tuple(manifest_entries))
    manifest_path = tmp_dir / "manifest.json"
    manifest.save(manifest_path)
    _write_done(tmp_dir / "DONE")

    _atomic_replace_dir(tmp_dir, final_dir)
    return final_dir / "manifest.json"


def build_epoch_plan(
    *,
    cfg: CacheProducerConfig,
    catalog: VideoCatalog,
    epoch: int,
    split: str,
) -> EpochPlan:
    from src.developing.mae.data.planning import make_bucket_specs

    bucket_specs = make_bucket_specs(
        cfg.buckets,
        alpha=cfg.bucket_alpha,
        base_batch_size=cfg.base_batch_size,
        min_batch_size=cfg.min_batch_size,
        patch_size=cfg.preprocess.patch_size,
    )
    return plan_epoch(
        catalog,
        epoch=epoch,
        split=split,
        samples_per_video=cfg.samples_per_video,
        buckets=bucket_specs,
        seed_base=cfg.seed,
        upsample_limit=cfg.upsample_limit,
        frame_sample_ratio=cfg.frame_sample_ratio,
        shuffle_batches=True,
    )


def ensure_current_pointer(cache_paths: EpochCachePaths, manifest_path: Path) -> None:
    cache_paths.split_dir().mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(str(manifest_path), start=str(cache_paths.split_dir()))
    cache_paths.current_pointer_path().write_text(rel + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    c = CacheProducerConfig(samples_per_video=2)
    root = Path(c.cache_root)
    root.mkdir(parents=True, exist_ok=True)
    print(asdict(c))

