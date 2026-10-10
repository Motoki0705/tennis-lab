"""Matched JPEG reader experiments with one timed verification phase. Queue only."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ball_mdd_query_gpu import measure  # type: ignore[import-not-found]

from src.tasks.ball_detection.data.coordinate_dataset import CoordinateWindowDataset
from src.tasks.ball_detection.data.store import shard_name
from src.tasks.ball_detection.training.coordinate_sampling import FPSMixSampler
from src.utils.checksum import dual_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=204)
    parser.add_argument("--io-stress", action="store_true", help="Compare deeper prefetch with selected-file cache eviction")
    args = parser.parse_args()
    if args.output.exists() or args.windows <= 12 or args.windows % 2:
        raise ValueError("Use a fresh output and even window count greater than 12")
    args.output.mkdir(parents=True)
    dataset = CoordinateWindowDataset(args.manifest, split="train", requires_pose=False, jpeg_decoder="nvjpeg")
    sequence = list(FPSMixSampler(dataset, windows_per_epoch=args.windows, seed=42))
    records = [dataset.records[i] for i in dict.fromkeys(dataset.windows[j][0] for j in sequence)]
    paths = [dataset.store.directory / "shards" / shard_name(record["clip"]["index"]) for record in records]
    print(json.dumps(dict(phase="verify_start", clips=len(paths))), flush=True)
    start = time.perf_counter()
    dataset._shared_rgb_verification.verify_all(8, paths=paths)
    setup = dict(seconds=time.perf_counter() - start, clips=len(paths), logical_bytes=sum(p.stat().st_size for p in paths),
                 workers=8, manifest_sha256=dual_sha256(args.manifest), queue_run_id=os.environ.get("TENNIS_RUN_ID"))
    (args.output / "verification.json").write_text(json.dumps(setup, indent=2) + "\n")
    print(json.dumps(dict(phase="verify_done", **setup)), flush=True)
    conditions = ([("async-w4-p8-b1", "pipeline", 4, 1, True, 8),
                   ("async-w8-p4-b1", "pipeline", 8, 1, True, 4),
                   ("async-prepared-b1", "prepared", 0, 1, True, 2)] if args.io_stress else
                  [("sync-w2-b1", "pipeline", 2, 1, False, 2),
                   ("async-w2-b1", "pipeline", 2, 1, True, 2),
                   ("async-w4-b1", "pipeline", 4, 1, True, 2),
                   ("async-w2-b2", "pipeline", 2, 2, True, 2),
                   ("async-prepared-b1", "prepared", 0, 1, True, 2)])
    for name, mode, workers, batch, prefetch, factor in conditions:
        torch._dynamo.reset()
        from torch._dynamo.utils import counters

        counters.clear()
        gc.collect()
        torch.cuda.empty_cache()
        options = argparse.Namespace(manifest=args.manifest, model_config=args.model_config, mode=mode,
            workers=workers, batch_size=batch, pin_memory=True, precision="bf16", compile_mode="default",
            cpu_threads=2, jpeg_decoder="nvjpeg", prefetch_factor=factor, image_prefetch=prefetch,
            preverify=False, sampler_windows=None, synchronize_steps=True, allocator_fraction=.9,
            drop_file_cache=args.io_stress and mode == "pipeline", warmup=12//batch, steps=(args.windows-12)//batch)
        report: dict[str, Any] = dict(condition=name, arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(options).items()},
                                    verification_shared=True, counters_reset_per_case=True)
        print(json.dumps(dict(phase="condition_start", condition=name)), flush=True)
        try:
            measure(options, report, dataset)
        except Exception as error:
            report.update(status="error", error=repr(error))
            raise
        finally:
            (args.output / f"{name}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(dict(condition=name, windows_per_second=report["windows_per_second"],
                             mean_loader_wait_seconds=report["mean_loader_wait_seconds"])), flush=True)


if __name__ == "__main__":
    main()
