"""Separate full-state resume checkpoints and weights-only fine-tuning artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from src.tasks.ball_detection.model_io.mdd_heatmap import (
    HeatmapModel,
    build_mdd_heatmap_detector,
    restore_heatmap_model,
)
from src.tasks.ball_detection.models.mdd import MDDHeatmapConfig
from src.utils.checksum import dual_sha256

RUN_SCHEMA = "ball_temporal_heatmap_run.v1"
CHECKPOINT_SCHEMA = "ball_temporal_heatmap_checkpoint.v1"


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def save_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_torch(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def rng_state(device: torch.device) -> dict[str, Any]:
    numpy = np.random.get_state(legacy=True)
    if not isinstance(numpy, tuple):
        raise ValueError("Expected NumPy's legacy tuple RNG state")
    return dict(torch=torch.get_rng_state(), python=random.getstate(),
                numpy=dict(algorithm=numpy[0], keys=numpy[1].tolist(), position=numpy[2],
                           has_gauss=numpy[3], cached_gaussian=numpy[4]),
                cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else None)


def restore_rng(saved: dict[str, Any], device: torch.device) -> None:
    if (saved["cuda"] is not None) != (device.type == "cuda"):
        raise ValueError("Resume cannot change CPU/CUDA RNG topology")
    torch.set_rng_state(saved["torch"])
    random.setstate(saved["python"])
    numpy = saved["numpy"]
    np.random.set_state((numpy["algorithm"], np.asarray(numpy["keys"], dtype=np.uint32), numpy["position"],
                         numpy["has_gauss"], numpy["cached_gaussian"]))
    if device.type == "cuda":
        if len(saved["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume CUDA device count changed")
        torch.cuda.set_rng_state_all(saved["cuda"])


def log_identity(path: Path) -> dict[str, Any]:
    return dict(bytes=path.stat().st_size, sha256=dual_sha256(path))


def restore_log(path: Path, identity: dict[str, Any], updates: int) -> None:
    with path.open("r+b") as stream:
        prefix = stream.read(identity["bytes"])
        if len(prefix) != identity["bytes"] or hashlib.sha256(prefix).hexdigest() != identity["sha256"]:
            raise ValueError("Committed train log changed or is incomplete")
        rows = [json.loads(line) for line in prefix.splitlines()]
        if [row["update"] for row in rows] != list(range(1, updates + 1)):
            raise ValueError("Committed train log is not a continuous update sequence")
        # Updates after the latest checkpoint were never committed; replay those draws.
        stream.truncate(identity["bytes"])


def initialize_finetuning(path: Path, *, expected_sha256: str, parent_update: int,
                          config: MDDHeatmapConfig) -> tuple[HeatmapModel, dict[str, Any]]:
    if dual_sha256(path) != expected_sha256:
        raise ValueError("Initial heatmap artifact hash mismatch")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    previous = restore_heatmap_model(saved)  # Validate the unmodified saved T first.
    training = saved.get("training", {})
    if (training.get("schema") != RUN_SCHEMA or training.get("phase") != "pretrain"
            or training.get("update") != parent_update or parent_update < 1 or previous.config.frames != 32):
        raise ValueError("Fine-tuning requires the specified update of a T=32 pretraining heatmap artifact")
    old, new = asdict(previous.config), asdict(config)
    for key in ("frames", "frame_chunk_size", "activation_checkpointing"):
        old.pop(key)
        new.pop(key)
    if type(previous.config) is not type(config) or old != new:
        raise ValueError("Fine-tuning must preserve architecture and every parameter topology")
    pair = build_mdd_heatmap_detector(config, mdd_a=previous.mdd.a, mdd_b=previous.mdd.b)
    model = cast(HeatmapModel, pair.model)
    model.load_state_dict(previous.state_dict(), strict=True)
    for key, value in previous.state_dict().items():
        other = model.state_dict()[key]
        if value.dtype != other.dtype or not torch.equal(value, other):
            raise ValueError(f"Strict full-state transfer failed: {key}")
    receipt = dict(mode="full_heatmap_weights", artifact_sha256=expected_sha256, update=parent_update,
                   saved_frames=32, target_frames=config.frames, training=training,
                   transferred_keys=sorted(previous.state_dict()))
    return model, receipt


def read_resume(path: Path, *, output: Path, recipe: dict[str, Any], code: dict[str, Any]) -> dict[str, Any]:
    latest = json.loads((output / "latest.json").read_text())
    if path.resolve() != (output / latest["checkpoint"]).resolve() or dual_sha256(path) != latest["sha256"]:
        raise ValueError("Resume requires this run's latest committed checkpoint and matching hash")
    saved: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("Resume requires an explicit temporal heatmap checkpoint, not a weights artifact")
    if json_bytes(saved["recipe"]) != json_bytes(recipe):
        raise ValueError("Resume recipe changed (including T, budget, data, initialization or runtime)")
    for key in ("source_sha256", "torch_version"):
        if saved["code"][key] != code[key]:
            raise ValueError("Resume code or torch version changed")
    update = saved["updates_completed"]
    budget = recipe["budget"]
    expected_schedule = dict(updates_completed=update, **budget)
    if saved["schedule"] != expected_schedule or saved["sample_position"] != update * (budget["input_frames"] // 128):
        raise ValueError("Resume schedule/update/pack position disagree")
    if latest["update"] != update or not 0 < update <= budget["updates"]:
        raise ValueError("Invalid saved update")
    return saved
