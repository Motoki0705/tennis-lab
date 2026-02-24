"""Epoch planning for cached-batch MAE training.

Produces an epoch plan that assigns exactly K samples per video while
controlling the resolution distribution via discrete buckets, ensuring
that each training batch contains a single resolution (no padding).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from src.experiments.mae.data.catalog import VideoCatalog, VideoMeta


@dataclass(frozen=True)
class BucketSpec:
    resolution: int
    weight: float
    batch_size: int


@dataclass(frozen=True)
class SampleSpec:
    video_id: int
    frame_idx: int
    resolution: int
    seed: int


@dataclass(frozen=True)
class BatchSpec:
    batch_id: int
    resolution: int
    samples: tuple[SampleSpec, ...]


@dataclass(frozen=True)
class EpochPlan:
    epoch: int
    split: str
    seed: int
    samples_per_video: int
    buckets: tuple[BucketSpec, ...]
    batches: tuple[BatchSpec, ...]

    def to_json(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "EpochPlan":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        buckets = tuple(BucketSpec(**b) for b in data["buckets"])
        batches = []
        for b in data["batches"]:
            samples = tuple(SampleSpec(**s) for s in b["samples"])
            batches.append(BatchSpec(batch_id=b["batch_id"], resolution=b["resolution"], samples=samples))
        return cls(
            epoch=int(data["epoch"]),
            split=str(data["split"]),
            seed=int(data["seed"]),
            samples_per_video=int(data["samples_per_video"]),
            buckets=buckets,
            batches=tuple(batches),
        )


def _round_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return max(multiple, (value // multiple) * multiple)


def make_bucket_specs(
    resolutions: Sequence[int],
    *,
    alpha: float,
    base_batch_size: int,
    min_batch_size: int,
    patch_size: int,
) -> tuple[BucketSpec, ...]:
    if not resolutions:
        raise ValueError("resolutions must be non-empty")
    if base_batch_size <= 0:
        raise ValueError("base_batch_size must be > 0")
    if min_batch_size <= 0:
        raise ValueError("min_batch_size must be > 0")

    resolutions_sorted = sorted({int(r) for r in resolutions})
    base_res = resolutions_sorted[0]
    specs: list[BucketSpec] = []
    for res in resolutions_sorted:
        res = _round_to_multiple(res, patch_size)
        weight = (base_res / res) ** float(alpha)
        bs = int(round(base_batch_size * (base_res / res) ** 2))
        bs = max(min_batch_size, bs)
        specs.append(BucketSpec(resolution=res, weight=float(weight), batch_size=int(bs)))
    return tuple(specs)


def _eligible_buckets(video: VideoMeta, buckets: Sequence[BucketSpec], upsample_limit: float) -> list[int]:
    limit = int(math.floor(video.short_side * float(upsample_limit)))
    eligible = [b.resolution for b in buckets if b.resolution <= limit]
    return eligible or [min(b.resolution for b in buckets)]


def _allocate_targets(
    *,
    total_samples: int,
    buckets: Sequence[BucketSpec],
    videos: Sequence[VideoMeta],
    samples_per_video: int,
    upsample_limit: float,
) -> dict[int, int]:
    capacities: dict[int, int] = {}
    for b in buckets:
        eligible_count = sum(
            1 for v in videos if b.resolution <= int(math.floor(v.short_side * float(upsample_limit)))
        )
        capacities[b.resolution] = eligible_count * samples_per_video

    weight_sum = sum(b.weight for b in buckets)
    if weight_sum <= 0:
        raise ValueError("Bucket weights must sum to > 0")

    targets: dict[int, int] = {}
    fractional: list[tuple[float, int]] = []
    assigned = 0
    for b in buckets:
        raw = total_samples * (b.weight / weight_sum)
        floor_val = int(math.floor(raw))
        targets[b.resolution] = floor_val
        assigned += floor_val
        fractional.append((raw - floor_val, b.resolution))

    remaining = total_samples - assigned
    for _, res in sorted(fractional, reverse=True):
        if remaining <= 0:
            break
        targets[res] += 1
        remaining -= 1

    overflow = True
    while overflow:
        overflow = False
        for b in buckets:
            res = b.resolution
            cap = capacities.get(res, 0)
            if targets[res] > cap:
                extra = targets[res] - cap
                targets[res] = cap
                overflow = True
                for b2 in buckets:
                    res2 = b2.resolution
                    if res2 == res:
                        continue
                    if res2 > res:
                        continue
                    cap2 = capacities.get(res2, 0)
                    room = cap2 - targets[res2]
                    if room <= 0:
                        continue
                    take = min(room, extra)
                    targets[res2] += take
                    extra -= take
                    if extra == 0:
                        break
                if extra != 0:
                    for b2 in buckets:
                        res2 = b2.resolution
                        if res2 == res:
                            continue
                        cap2 = capacities.get(res2, 0)
                        room = cap2 - targets[res2]
                        if room <= 0:
                            continue
                        take = min(room, extra)
                        targets[res2] += take
                        extra -= take
                        if extra == 0:
                            break
                if extra != 0:
                    raise RuntimeError("Unable to allocate samples within bucket capacities.")

    if sum(targets.values()) != total_samples:
        raise RuntimeError("Target allocation failed to match total_samples.")

    return targets


def plan_epoch(
    catalog: VideoCatalog,
    *,
    epoch: int,
    split: str,
    samples_per_video: int,
    buckets: Sequence[BucketSpec],
    seed_base: int,
    upsample_limit: float = 1.0,
    frame_sample_ratio: float = 1.0,
    shuffle_batches: bool = True,
) -> EpochPlan:
    videos = list(catalog.videos)
    if not videos:
        raise ValueError("catalog must contain at least one video")
    if samples_per_video <= 0:
        raise ValueError("samples_per_video must be > 0")
    if not buckets:
        raise ValueError("buckets must be non-empty")
    if not (0 < frame_sample_ratio <= 1.0):
        raise ValueError("frame_sample_ratio must be in (0, 1]")

    seed = int(seed_base + epoch)
    rng = random.Random(seed)
    total_samples = len(videos) * samples_per_video

    targets = _allocate_targets(
        total_samples=total_samples,
        buckets=buckets,
        videos=videos,
        samples_per_video=samples_per_video,
        upsample_limit=upsample_limit,
    )

    eligible_cache: dict[int, list[int]] = {}
    for v in videos:
        eligible_cache[v.video_id] = _eligible_buckets(v, buckets, upsample_limit)

    remaining_by_bucket = {k: int(v) for k, v in targets.items()}

    sample_specs: list[SampleSpec] = []
    for v in videos:
        eligible = eligible_cache[v.video_id]
        for k in range(samples_per_video):
            candidates = sorted(
                eligible,
                key=lambda r: (-remaining_by_bucket.get(r, 0), r),
            )
            chosen = candidates[0]
            if remaining_by_bucket.get(chosen, 0) > 0:
                remaining_by_bucket[chosen] -= 1
            else:
                chosen = min(eligible)

            n = v.num_frames
            if n <= 0:
                frame_idx = 0
            else:
                usable = max(1, int(math.floor(n * frame_sample_ratio)))
                start = n - usable
                frame_idx = rng.randint(start, n - 1)

            sample_seed = rng.randint(0, 2**31 - 1)
            sample_specs.append(
                SampleSpec(
                    video_id=v.video_id,
                    frame_idx=int(frame_idx),
                    resolution=int(chosen),
                    seed=int(sample_seed),
                )
            )

    if shuffle_batches:
        rng.shuffle(sample_specs)

    samples_by_res: dict[int, list[SampleSpec]] = {}
    bucket_bs = {b.resolution: b.batch_size for b in buckets}
    for s in sample_specs:
        samples_by_res.setdefault(s.resolution, []).append(s)

    batches: list[BatchSpec] = []
    batch_id = 0
    for res in sorted(samples_by_res.keys()):
        bs = int(bucket_bs.get(res, 1))
        samples = samples_by_res[res]
        for i in range(0, len(samples), bs):
            chunk = tuple(samples[i : i + bs])
            batches.append(BatchSpec(batch_id=batch_id, resolution=int(res), samples=chunk))
            batch_id += 1

    if shuffle_batches:
        rng.shuffle(batches)
        batches = [BatchSpec(batch_id=i, resolution=b.resolution, samples=b.samples) for i, b in enumerate(batches)]

    return EpochPlan(
        epoch=int(epoch),
        split=str(split),
        seed=int(seed),
        samples_per_video=int(samples_per_video),
        buckets=tuple(buckets),
        batches=tuple(batches),
    )


def split_video_paths(
    catalog: VideoCatalog,
    *,
    val_split: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    if not (0.0 <= val_split < 1.0):
        raise ValueError("val_split must be in [0, 1)")
    paths = [v.path for v in catalog.videos]
    rng = random.Random(int(seed))
    rng.shuffle(paths)
    num_val = int(len(paths) * val_split)
    val_paths = paths[:num_val]
    train_paths = paths[num_val:]
    return train_paths, val_paths


def save_path_list(paths: Iterable[str], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(list(paths), indent=2), encoding="utf-8")


def load_path_list(path: str | Path) -> list[str]:
    return list(json.loads(Path(path).read_text(encoding="utf-8")))


if __name__ == "__main__":  # pragma: no cover
    dummy = VideoCatalog(
        videos=(
            VideoMeta(video_id=0, path="v0.mp4", num_frames=300, width=1920, height=1080),
            VideoMeta(video_id=1, path="v1.mp4", num_frames=120, width=1280, height=720),
            VideoMeta(video_id=2, path="v2.mp4", num_frames=80, width=640, height=360),
        )
    )
    bucket_specs = make_bucket_specs(
        [256, 320, 384, 512, 768, 1024],
        alpha=2.0,
        base_batch_size=32,
        min_batch_size=1,
        patch_size=16,
    )
    plan = plan_epoch(
        dummy,
        epoch=0,
        split="train",
        samples_per_video=4,
        buckets=bucket_specs,
        seed_base=42,
        upsample_limit=1.0,
    )
    print(f"batches={len(plan.batches)} samples={sum(len(b.samples) for b in plan.batches)}")

