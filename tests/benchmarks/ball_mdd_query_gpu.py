"""Real-input training throughput diagnostic. Run CUDA only via training queue.

Each invocation measures one configuration in its own process. No validation or
test labels are used, and no trained checkpoint is retained. Compute mode keeps
one real RGB batch resident; pipeline mode includes the JPEG/uint8 reader.
Fixed FP32 MDD is part of the model in both modes. BF16 is used when requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import torch

from src.tasks.ball_detection.data.coordinate_dataset import (
    CoordinateWindowDataset,
    collate_coordinate_windows,
)
from src.tasks.ball_detection.models.mdd_pose import MDDPoseConfig, MDDQueryDetector
from src.tasks.ball_detection.training.coordinate_compilation import (
    COMPILE_MODES,
    coordinate_compilation_report,
    coordinate_compile_scope,
)
from src.tasks.ball_detection.training.coordinate_evaluation import predict_coordinates
from src.tasks.ball_detection.training.coordinate_runtime import (
    CoordinateRuntime,
    coordinate_train_step,
)
from src.tasks.ball_detection.training.coordinate_sampling import FPSMixSampler


def chosen_indices(dataset: CoordinateWindowDataset, count: int) -> list[int]:
    groups: dict[tuple[str, int], list[int]] = {}
    for index, (record, window) in enumerate(dataset.windows):
        source = dataset.records[record]["clip"]["source"]
        groups.setdefault((source, window.frame_step), []).append(index)
    keys = sorted(groups)
    return [groups[keys[i % len(keys)]][i // len(keys)] for i in range(count)]


def measure(args: argparse.Namespace, report: dict[str, Any]) -> None:
    torch.set_num_threads(args.cpu_threads)
    cv2.setNumThreads(1)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(.9)
    properties = torch.cuda.get_device_properties(device)
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 is not supported; no precision fallback")
    report["hardware"] = dict(gpu=properties.name, total_bytes=properties.total_memory,
                              cuda=torch.version.cuda, torch=str(torch.__version__),
                              python=platform.python_version(), memory_fraction_limit=.9,
                              free_before_bytes=torch.cuda.mem_get_info()[0])
    config = MDDPoseConfig.load(args.model_config)
    if config.compression != "conv2d" or config.readout != "query_only":
        raise ValueError("This benchmark is specifically Conv2d + query-only")
    report["model"] = asdict(config)
    dataset = CoordinateWindowDataset(args.manifest, split="train", requires_pose=False)
    model = MDDQueryDetector(config).to(device).train()
    runtime = CoordinateRuntime(precision=args.precision, num_workers=args.workers,
                                pin_memory=args.pin_memory, cpu_threads=args.cpu_threads,
                                compile_mode=args.compile_mode)
    runtime.configure(device)
    runtime.configure_model(model)
    report["input_contract"] = model.mdd.input_contract()
    report["parameters"] = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    first_parameter = next(model.parameters()).detach().clone()
    total = args.warmup + args.steps
    if args.mode == "compute":
        indices = chosen_indices(dataset, args.batch_size)
        batch = collate_coordinate_windows([dataset[i] for i in indices])
        report["samples"] = [dict(clip_id=c, start=s, frame_step=f) for c, s, f in
                             zip(batch["clip_id"], batch["start"], batch["frame_step"], strict=True)]
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        iterator = iter([batch] * total)
    else:
        sampler = FPSMixSampler(dataset, windows_per_epoch=total * args.batch_size, seed=42)
        loader = runtime.loader(dataset, batch_size=args.batch_size, sampler=sampler, seed=42)
        report["window_sequence_sha256"] = hashlib.sha256(json.dumps(list(sampler)).encode()).hexdigest()
        iterator = iter(loader)
    report["steps"] = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for step in range(total):
        begin = time.perf_counter()
        batch = next(iterator)
        loaded = time.perf_counter()
        loss, grad = coordinate_train_step(model, optimizer, batch, device, runtime)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        row = dict(step=step, warmup=step < args.warmup, seconds=elapsed,
                   load_wait_seconds=loaded - begin, loss=float(loss.detach()), grad_norm=float(grad))
        report["steps"].append(row)
        if step == 0 or step + 1 == total or (step + 1) % 10 == 0:
            print(json.dumps(row), flush=True)
    measured = report["steps"][args.warmup:]
    seconds = sum(row["seconds"] for row in measured)
    report.update(status="ok", windows_per_second=args.batch_size * args.steps / seconds,
                  sampled_frames_per_second=32 * args.batch_size * args.steps / seconds,
                  mean_step_seconds=seconds / args.steps,
                  median_step_seconds=statistics.median(row["seconds"] for row in measured),
                  mean_loader_wait_seconds=statistics.mean(row["load_wait_seconds"] for row in measured),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  final_loss=report["steps"][-1]["loss"],
                  parameter_max_change=float((next(model.parameters()).detach() - first_parameter).abs().max()))
    model.eval()
    with torch.no_grad(), coordinate_compile_scope(model):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
            output = predict_coordinates(model, batch, device)
        report["eval_finite"] = bool(torch.isfinite(output).all())
    report["compilation"] = coordinate_compilation_report(model)
    if args.compile_mode != "off":
        assert report["compilation"]["unique_graphs"] > 0
        assert not report["compilation"]["graph_breaks"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--precision", choices=("fp32", "bf16"), required=True)
    parser.add_argument("--compile-mode", choices=COMPILE_MODES, default="off")
    parser.add_argument("--mode", choices=("compute", "pipeline"), default="compute")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a fresh output file")
    if min(args.batch_size, args.cpu_threads, args.steps) < 1 or min(args.warmup, args.workers) < 0:
        raise ValueError("Invalid measurement budget")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = dict(schema="ball_native_rgb_gpu.v2", status="running",
                                 arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                 manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                                 model_config_sha256=hashlib.sha256(args.model_config.read_bytes()).hexdigest(),
                                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                 commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                                 queue_run_id=os.environ.get("TENNIS_RUN_ID"), seed=42,
                                 optimizer=dict(name="AdamW", lr=1e-4, weight_decay=.01),
                                 tf32=dict(matmul=False, cudnn=True), cudnn_benchmark=False,
                                 train_only=True, checkpoint_saved=False)
    try:
        measure(args, report)
    except torch.cuda.OutOfMemoryError as error:
        report.update(status="oom", error=str(error),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as error:
        report.update(status="error", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"steps", "samples", "model"}}), flush=True)


if __name__ == "__main__":
    main()
