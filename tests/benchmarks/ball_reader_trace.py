"""Reproduce slow real windows and time worker, page faults, IPC and pinning."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import torch
import torch.utils.data._utils.pin_memory as pin_module

import src.tasks.ball_detection.training.coordinate_runtime as runtime_module
import src.utils.shared_file_verification as verification
from src.tasks.ball_detection.data.coordinate_dataset import (
    CoordinateWindowDataset,
    collate_coordinate_windows,
)
from src.tasks.ball_detection.data.store import shard_name
from src.tasks.ball_detection.models.mdd_pose import MDDPoseConfig, MDDQueryDetector
from src.tasks.ball_detection.training.coordinate_images import coordinate_batches
from src.tasks.ball_detection.training.coordinate_runtime import (
    CoordinateRuntime,
    coordinate_train_step,
)
from src.tasks.ball_detection.training.coordinate_sampling import FPSMixSampler

CURRENT: dict[str, Any] | None = None


def instrument(function: Any, name: str) -> Any:
    def call(*args: Any, **kwargs: Any) -> Any:
        start, cpu = time.perf_counter(), time.process_time()
        result = function(*args, **kwargs)
        if CURRENT is not None:
            CURRENT[name + "_seconds"] = CURRENT.get(name + "_seconds", 0.) + time.perf_counter() - start
            CURRENT[name + "_cpu_seconds"] = CURRENT.get(name + "_cpu_seconds", 0.) + time.process_time() - cpu
        return result
    return call


class TimedDataset(CoordinateWindowDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        global CURRENT
        CURRENT = dict(index=index, pid=os.getpid())
        start, cpu, before = time.perf_counter(), time.process_time(), resource.getrusage(resource.RUSAGE_SELF)
        sample: dict[str, Any] = super().__getitem__(index)
        after = resource.getrusage(resource.RUSAGE_SELF)
        sample["_worker"] = dict(CURRENT, seconds=time.perf_counter()-start,
            cpu_seconds=time.process_time()-cpu, major_faults=after.ru_majflt-before.ru_majflt,
            block_reads=after.ru_inblock-before.ru_inblock, jpeg_bytes=sample["jpeg"].numel())
        CURRENT = None
        return sample


def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    begin, cpu = time.perf_counter(), time.process_time()
    result: dict[str, Any] = collate_coordinate_windows(samples)
    result.update(_worker=[s["_worker"] for s in samples], _collate_seconds=time.perf_counter()-begin,
                  _collate_cpu_seconds=time.process_time()-cpu)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sustained-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose fresh trace output")
    prior = json.loads(args.sustained_report.read_text())
    data = TimedDataset(Path(prior["arguments"]["manifest"]), split="train", requires_pose=False, jpeg_decoder="nvjpeg")
    sequence = list(FPSMixSampler(data, windows_per_epoch=len(prior["steps"]), seed=42))
    assert hashlib.sha256(json.dumps(sequence).encode()).hexdigest() == prior["window_sequence_sha256"]
    slow = sorted([r for r in prior["steps"] if not r["warmup"]], key=lambda r: r["reader_wait_seconds"], reverse=True)[:8]
    selected = [sequence[r["step"]] for r in slow]
    paths = list(dict.fromkeys(data.store.directory / "shards" / shard_name(data.records[data.windows[i][0]]["clip"]["index"]) for i in selected))
    data._shared_rgb_verification.verify_all(2, paths=paths)
    # This is only an OS hint to evict clean pages of these eight input files.
    # It changes no file contents/stat identity and does not promise a cold disk.
    for path in paths:
        with path.open("rb") as stream:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    verification.dual_sha256 = instrument(verification.dual_sha256, "hash")
    TimedDataset._read_images = instrument(TimedDataset._read_images, "images")  # type: ignore[method-assign]
    TimedDataset._verify_clip = instrument(TimedDataset._verify_clip, "verify")  # type: ignore[method-assign]
    runtime_module.collate_coordinate_windows = collate
    original_pin = pin_module.pin_memory

    def timed_pin(value: Any, device: Any = None) -> Any:
        start, cpu = time.perf_counter(), time.process_time()
        result = original_pin(value, device)
        if isinstance(result, dict) and "jpeg" in result:
            result.update(_pin_seconds=time.perf_counter()-start, _pin_cpu_seconds=time.process_time()-cpu)
        return result

    pin_module.pin_memory = timed_pin
    runtime = CoordinateRuntime(precision="bf16", compile_mode="default", jpeg_decoder="nvjpeg",
                                image_prefetch=True, num_workers=2, pin_memory=True, prefetch_factor=2)
    device = torch.device("cuda")
    runtime.configure(device)
    model = MDDQueryDetector(MDDPoseConfig.load(Path(prior["arguments"]["model_config"]))).to(device).train()
    runtime.configure_model(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loader = runtime.loader(data, batch_size=1, sampler=selected * 4)
    rows = []
    iterator = coordinate_batches(loader, device, prefetch=True)
    for i, batch in enumerate(iterator):
        loss, _ = coordinate_train_step(model, optimizer, batch, device, runtime)
        torch.cuda.synchronize()
        row = dict(step=i, cycle=i//8, worker=batch["_worker"][0], pin_seconds=batch["_pin_seconds"],
                   pin_cpu_seconds=batch["_pin_cpu_seconds"], collate_seconds=batch["_collate_seconds"],
                   collate_cpu_seconds=batch["_collate_cpu_seconds"], reader_wait_seconds=batch["_reader_wait_seconds"],
                   image_prepare_seconds=batch["_image_prepare_seconds"], loss=float(loss))
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(status="ok", slow_steps=[r["step"] for r in slow], selected_indices=selected,
        cache_hint="POSIX_FADV_DONTNEED once per selected file before reading", rows=rows), indent=2)+'\n')


if __name__ == "__main__":
    main()
