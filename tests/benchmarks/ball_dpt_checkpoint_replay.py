"""Queue-only, bounded replay of an epoch checkpoint; never saves training weights."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from itertools import islice
from pathlib import Path
from typing import Any

import torch

from src.tasks.ball_detection.data.coordinate_dataset import collate_coordinate_windows
from src.tasks.ball_detection.models.mdd_pretrain import MDDDPTDetector
from src.tasks.ball_detection.training.coordinate_compilation import (
    compile_coordinate_model,
    coordinate_compile_scope,
)
from src.tasks.ball_detection.training.coordinate_runtime import CoordinateRuntime
from src.tasks.ball_detection.training.coordinate_sampling import FPSMixSampler
from src.tasks.ball_detection.training.heatmap_pretraining.checkpoint import (
    pretraining_config,
)
from src.tasks.ball_detection.training.heatmap_pretraining.data import (
    HeatmapWindowDataset,
)
from src.tasks.ball_detection.training.heatmap_pretraining.diagnostics import (
    batch_identity,
    record_failure,
)
from src.tasks.ball_detection.training.heatmap_pretraining.objective import (
    heatmap_objective,
    image_input,
)
from src.tasks.ball_detection.training.heatmap_pretraining.runner import (
    learning_rate,
    source_identity,
)
from src.utils.checksum import dual_sha256


def verify_target(batches: list[dict[str, Any]], *, first_step: int, failure: dict[str, Any]) -> None:
    offset = failure["attempted_update"] - first_step - 1
    if not 0 <= offset < len(batches):
        raise ValueError("Replay does not include the failed update")
    if batch_identity(batches[offset]) != failure["batch"]:
        raise ValueError("Replayed sampler prefix does not match the failed batch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "checkpoint", "failure", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--updates", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.updates <= 128:
        parser.error("The diagnostic is bounded to 1..128 updates")
    if os.environ.get("CUDA_LAUNCH_BLOCKING") != "1" or not os.environ.get("TENNIS_RUN_ID"):
        parser.error("Use the training queue with CUDA_LAUNCH_BLOCKING=1")
    if any(not getattr(args, key).is_absolute() for key in ("manifest", "checkpoint", "failure", "output")):
        parser.error("All paths must be absolute")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    recipe = saved["recipe"]
    if saved["code"]["source_sha256"] != source_identity()["source_sha256"]:
        raise ValueError("Checkpoint training implementation changed")
    if recipe["manifest_sha256"] != dual_sha256(args.manifest):
        raise ValueError("Frozen manifest changed")
    if recipe["runtime"]["image_prefetch"] or recipe["runtime"]["precision"] != "bf16" or recipe["batch_size"] != 1:
        raise ValueError("This diagnostic requires serial BF16 BS1 pretraining")
    epoch, step = saved["epoch"] + 1, saved["global_step"]
    if step != epoch * recipe["windows_per_epoch"]:
        raise ValueError("Expected a completed epoch checkpoint")
    data = HeatmapWindowDataset(args.manifest, split="train", jpeg_decoder=recipe["runtime"]["jpeg_decoder"])
    sampler = FPSMixSampler(data, windows_per_epoch=recipe["windows_per_epoch"], seed=recipe["seed"])
    sampler.set_epoch(epoch)
    indices = list(islice(iter(sampler), args.updates))
    # __getitem__ verifies each selected shard and semantic hash on CPU before CUDA startup.
    # Preloading changes only input scheduling, not the frozen sampler/bytes/GT.
    batches = [collate_coordinate_windows([data[index]]) for index in indices]
    failure = json.loads(args.failure.read_text())
    verify_target(batches, first_step=step, failure=failure)
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=dual_sha256(args.checkpoint),
                   manifest_sha256=recipe["manifest_sha256"], first_step=step, updates=args.updates,
                   target_attempt=failure["attempted_update"], sampler_indices=indices,
                   precision="bf16", launch_blocking=True, phase_synchronization=True,
                   input_scheduling="verified JPEG prefix preloaded on CPU; no DataLoader workers",
                   modifies_original_run=False, saves_training_checkpoint=False)
    (args.output / "config.json").write_text(json.dumps(receipt, indent=2))
    runtime = CoordinateRuntime(**recipe["runtime"])
    device = torch.device("cuda")
    runtime.configure(device)
    torch.manual_seed(recipe["seed"])
    model = MDDDPTDetector(pretraining_config(saved)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    model.load_state_dict(saved["state_dict"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])
    compile_coordinate_model(model, mode=runtime.compile_mode, recompile_limit=runtime.compile_recompile_limit)
    model.train()
    context: dict[str, Any] = dict(epoch=epoch, completed_updates=step)
    started = time.perf_counter()
    with record_failure(args.output, context):
        for batch in batches:
            context.update(attempted_update=step + 1, batch=batch_identity(batch), surface_phase="decode")
            rgb = image_input(batch, device)
            torch.cuda.synchronize()
            lr = learning_rate(step, peak=recipe["learning_rate"], total=recipe["epochs"] * recipe["windows_per_epoch"],
                               warmup=recipe["warmup_updates"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with coordinate_compile_scope(model):
                context["surface_phase"] = "forward"
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(rgb)
                torch.cuda.synchronize()
                context["surface_phase"] = "loss"
                loss, _ = heatmap_objective(logits, batch, sigma_ratio=recipe["sigma_ratio"], gamma=recipe["focal_gamma"])
                torch.cuda.synchronize()
                context["surface_phase"] = "backward"
                loss.backward()
                torch.cuda.synchronize()
            context["surface_phase"] = "gradient_norm"
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["gradient_clip"], error_if_nonfinite=True)
            torch.cuda.synchronize()
            context["surface_phase"] = "optimizer"
            optimizer.step()
            torch.cuda.synchronize()
            value = float(loss.detach())
            if not math.isfinite(value):
                raise ValueError("Nonfinite diagnostic loss")
            step += 1
            context["completed_updates"] = step
            row = dict(global_step=step, loss=value, grad_norm=float(grad), learning_rate=lr, batch=context["batch"])
            with (args.output / "steps.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    receipt.update(last_step=step, seconds=time.perf_counter() - started,
                   limitation="Short synchronized replay does not establish long-run stability or identify the original cause")
    (args.output / "PROBE_COMPLETED.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
