"""CPU-only diagnosis of the unchanged coordinate reader; never starts training.

Instrumentation delegates to production functions. The preverified case performs
the same complete integrity checks in the parent before forking (timed separately).
The cached-input case reuses one real sample to isolate collation and CPU IPC.
There is no CUDA, pinning, H2D transfer, model, backward pass, or data mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import resource
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

import src.tasks.ball_detection.data.coordinate_dataset as reader
from src.tasks.ball_detection.data.store import BallFrameStore, shard_name
from src.tasks.ball_detection.training.coordinate_sampling import FPSMixSampler
from src.utils.checksum import dual_sha256

CURRENT: dict[str, Any] | None = None


def instrument(function: Any, key: str) -> Any:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        start, cpu = time.perf_counter(), time.process_time()
        result = function(*args, **kwargs)
        if CURRENT is not None:
            CURRENT[key + "_seconds"] = CURRENT.get(key + "_seconds", 0.) + time.perf_counter() - start
            CURRENT[key + "_cpu_seconds"] = CURRENT.get(key + "_cpu_seconds", 0.) + time.process_time() - cpu
        return result
    return wrapped


class TimedDataset(reader.CoordinateWindowDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        global CURRENT
        record_index, window = self.windows[index]
        record = self.records[record_index]
        clip_id = record["clip"]["clip_id"]
        worker = get_worker_info()
        CURRENT = dict(index=index, clip_id=clip_id, frame_step=window.frame_step,
                       worker=-1 if worker is None else worker.id, pid=os.getpid(),
                       hash_miss=clip_id not in self._rgb_versions,
                       shard_bytes=(self.store.directory / "shards" / shard_name(record["clip"]["index"])).stat().st_size)
        start, cpu = time.perf_counter(), time.process_time()
        sample = super().__getitem__(index)
        CURRENT.update(prepare_seconds=time.perf_counter() - start,
                       prepare_cpu_seconds=time.process_time() - cpu)
        sample["_profile"] = CURRENT
        CURRENT = None
        return sample


class CachedInput(Dataset[dict[str, Any]]):
    def __init__(self, sample: dict[str, Any], length: int) -> None:
        self.sample, self.length = sample, length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        worker = get_worker_info()
        return dict(self.sample, _profile=dict(index=index, worker=-1 if worker is None else worker.id,
                                               pid=os.getpid(), hash_miss=False,
                                               prepare_seconds=0., prepare_cpu_seconds=0.))


def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    start, cpu = time.perf_counter(), time.process_time()
    result = reader.collate_coordinate_windows(samples)
    result["_profiles"] = [s["_profile"] for s in samples]
    result["_collate_seconds"] = time.perf_counter() - start
    result["_collate_cpu_seconds"] = time.process_time() - cpu
    result["_collate_finished"] = time.perf_counter()
    return result


def worker_init(_: int) -> None:
    torch.set_num_threads(1)
    cv2.setNumThreads(1)


def process_io() -> dict[str, int]:
    fields = {}
    for line in Path("/proc/self/io").read_text().splitlines():
        key, value = line.split(":")
        fields[key] = int(value)
    return fields


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=("normal", "preverified", "cached_input"), required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--windows", type=int, default=96)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output already exists")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Explicitly hide CUDA devices for this CPU-only diagnostic")
    if multiprocessing.get_start_method() != "fork":
        raise ValueError("This diagnostic measures Linux fork workers with inherited caches")
    worker_init(-1)
    reader.luminance_to_mdd = instrument(reader.luminance_to_mdd, "mdd")
    reader.dual_sha256 = instrument(reader.dual_sha256, "hash")
    reader.CoordinateWindowDataset._verify_clip = instrument(reader.CoordinateWindowDataset._verify_clip, "verify")
    BallFrameStore.read_bgr = instrument(BallFrameStore.read_bgr, "read_bgr")
    BallFrameStore.read_jpeg = instrument(BallFrameStore.read_jpeg, "jpeg_bytes")
    setup_start = time.perf_counter()
    data = TimedDataset(args.manifest, split="train", requires_pose=False, mdd_a=.2, mdd_b=.15)
    indices = list(FPSMixSampler(data, windows_per_epoch=6000, seed=42))[:args.windows]
    report: dict[str, Any] = dict(schema="mdd_cpu_profile.v1", case=args.case, workers=args.workers,
        windows=args.windows, torch_threads_per_process=1, opencv_threads=1,
        manifest_sha256=dual_sha256(args.manifest), source_commit="0008c6d0b6123ce1c555523a4de6133255cc5151",
        window_sequence_sha256=hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
        pin_memory=False, device="cpu", prefetch_factor=1, model_executed=False,
        setup_seconds=time.perf_counter() - setup_start,
        caveat="first verification does not imply a cold OS page cache; no GPU/H2D/pinned-memory measurement")
    if args.case == "preverified":
        start = time.perf_counter()
        initial_io = process_io()
        unique = dict.fromkeys(data.windows[i][0] for i in indices)
        for record_index in unique:
            record = data.records[record_index]
            data._verify_clip(record, data.store.clip_by_id(record["clip"]["clip_id"]))
        report["preverification"] = dict(seconds=time.perf_counter() - start, clips=len(unique),
                                        process_io_delta={k: v - initial_io[k] for k, v in process_io().items()})
    dataset: Dataset[dict[str, Any]] = data
    if args.case == "cached_input":
        sample = data[indices[0]]
        dataset = CachedInput(sample, args.windows)
        indices = list(range(args.windows))
    loader = DataLoader(dataset, batch_size=1, sampler=indices, num_workers=args.workers,
                        pin_memory=False, prefetch_factor=1 if args.workers else None,
                        persistent_workers=args.workers > 0, worker_init_fn=worker_init, collate_fn=collate)
    rounds = []
    for repeat in range(2):
        start = time.perf_counter()
        previous = start
        rows = []
        for batch in loader:
            arrived = time.perf_counter()
            row = dict(batch["_profiles"][0], collate_seconds=batch["_collate_seconds"],
                       collate_cpu_seconds=batch["_collate_cpu_seconds"],
                       consumer_wait_seconds=arrived - previous,
                       after_collate_seconds=arrived - batch["_collate_finished"])
            rows.append(row)
            previous = time.perf_counter()
        seconds = time.perf_counter() - start
        # Stage sums are worker service time, not additive critical-path time:
        # workers overlap and arrival delay includes queue residence and IPC.
        keys = ("prepare", "verify", "hash", "read_bgr", "jpeg_bytes", "mdd", "collate")
        stage_means = {key: statistics.mean(r.get(key + "_seconds", 0.) for r in rows) for key in keys}
        stage_means["luminance_and_metadata"] = stage_means["prepare"] - stage_means["verify"] - stage_means["read_bgr"] - stage_means["mdd"]
        result = dict(repeat=repeat, seconds=seconds, windows_per_second=len(rows) / seconds,
                      first_batch_seconds=rows[0]["consumer_wait_seconds"],
                      without_first_batch_windows_per_second=(len(rows)-1)/(seconds-rows[0]["consumer_wait_seconds"]),
                      hash_misses=sum(r["hash_miss"] for r in rows),
                      hash_logical_bytes=sum(r.get("shard_bytes", 0) for r in rows if r["hash_miss"]),
                      mean_stage_seconds=stage_means,
                      mean_prepare_cpu_seconds=statistics.mean(r["prepare_cpu_seconds"] for r in rows),
                      mean_after_collate_seconds=statistics.mean(r["after_collate_seconds"] for r in rows),
                      rows=rows)
        rounds.append(result)
        report["rounds"] = rounds
        report["rss_max_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
