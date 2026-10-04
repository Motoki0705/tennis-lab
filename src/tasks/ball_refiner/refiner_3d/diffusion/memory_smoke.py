"""Bounded memory/gradient diagnostic; no dataset training or accuracy claim."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from src.tasks.ball_refiner.refiner_3d.diffusion.flow import training_objective
from src.tasks.ball_refiner.refiner_3d.diffusion.losses import LossConfig
from src.tasks.ball_refiner.refiner_3d.diffusion.memory_fixture import (
    analytic_memory_batch,
)
from src.tasks.ball_refiner.refiner_3d.diffusion.model import (
    ModelConfig,
    TrajectoryDenoiser,
)
from src.tasks.ball_refiner.refiner_3d.synthetic.configuration import sha256
from src.tasks.ball_refiner.refiner_3d.synthetic.generator import write_json
from src.utils.schema.court_normalization import court_coordinate_normalization_metadata


def run_memory_smoke(config_path: Path, fixture: Path, output: Path, *, device: str) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    raw = yaml.safe_load(config_path.read_text())
    required = {"input_kind", "seed", "updates", "batch_size", "frames", "learning_rate", "maximum_seconds", "allocator_limit_gib", "model", "loss"}
    if not isinstance(raw, dict) or set(raw) != required or raw["input_kind"] != "analytic_memory_fixture_v1":
        raise ValueError("Need the complete analytic memory diagnostic configuration")
    for key in ("seed", "updates", "batch_size", "frames", "maximum_seconds"):
        if type(raw[key]) is not int or raw[key] < 1:
            raise ValueError(f"Need a positive integer {key}")
    if raw["updates"] != 100 or raw["maximum_seconds"] > 780 or not 0 < raw["allocator_limit_gib"] <= 4:
        raise ValueError("Memory diagnostic exceeds the 100-update/time/allocator grant")
    if not math.isfinite(raw["learning_rate"]) or raw["learning_rate"] <= 0:
        raise ValueError("Invalid learning rate")
    if device not in ("cpu", "cuda"):
        raise ValueError("Device must be explicitly cpu or cuda")
    model_config, loss_config = ModelConfig(**raw["model"]), LossConfig(**raw["loss"])
    torch.set_num_threads(1)
    torch.manual_seed(raw["seed"])
    started = time.perf_counter()
    device_memory_peak: int | None = None
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., raw["allocator_limit_gib"] * 1024 ** 3 / total), 0)
        torch.cuda.reset_peak_memory_stats()
        free, capacity = torch.cuda.mem_get_info()
        device_memory_peak = capacity - free
        if device_memory_peak > 6_000_000_000:
            raise RuntimeError("Device memory already exceeds the 6 GB smoke budget")
    batch = analytic_memory_batch(fixture, batch_size=raw["batch_size"], frames=raw["frames"], seed=raw["seed"]).to(device)
    model = TrajectoryDenoiser(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=raw["learning_rate"])
    generator = torch.Generator(device=device).manual_seed(raw["seed"] + 1)
    manifest: dict[str, Any] = {
        "status": "running", "diagnostic_only": True, "device": device, "config": raw,
        "config_sha256": sha256(config_path), "fixture_sha256": sha256(fixture),
        "parameters": sum(p.numel() for p in model.parameters()),
        "court_coordinate_normalization": court_coordinate_normalization_metadata(),
        "dataset_status": "not_used; analytic tensors only; synthetic rally generator remains a separate gate",
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", manifest)
    try:
        with (output / "updates.jsonl").open("w") as log:
            for update in range(1, raw["updates"] + 1):
                if time.perf_counter() - started > raw["maximum_seconds"]:
                    raise TimeoutError("Memory smoke exceeded its wall-clock budget")
                optimizer.zero_grad(set_to_none=True)
                loss, terms = training_objective(model, batch, loss_config, generator, objective="flow")
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite loss")
                loss.backward()
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise FloatingPointError("Nonfinite gradients")
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                if device == "cuda":
                    torch.cuda.synchronize()
                    free, capacity = torch.cuda.mem_get_info()
                    assert device_memory_peak is not None
                    device_memory_peak = max(device_memory_peak, capacity - free)
                    if device_memory_peak > 6_000_000_000:
                        raise RuntimeError("Device memory exceeded the 6 GB smoke budget")
                row = {"update": update, "loss": loss.item(), "gradient_norm": norm.item(), **{key: value.item() for key, value in terms.items()}}
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
        checkpoint = output / "diagnostic-only.pt"
        torch.save({"diagnostic_only": True, "model": asdict(model_config), "state_dict": model.state_dict(), "updates": raw["updates"]}, checkpoint)
        manifest.update(
            status="complete", updates=raw["updates"], elapsed_seconds=time.perf_counter() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if device == "cuda" else None,
            peak_device_used_bytes=device_memory_peak,
            device_memory_note="CUDA driver total-minus-free, sampled per update; includes unrelated device usage and can miss between-update peaks",
            checkpoint_bytes=checkpoint.stat().st_size, checkpoint_sha256=sha256(checkpoint),
        )
    except Exception as exc:
        manifest.update(status="failed", error=str(exc), elapsed_seconds=time.perf_counter() - started)
        write_json(output / "manifest.json", manifest)
        raise
    write_json(output / "manifest.json", manifest)
    return manifest
