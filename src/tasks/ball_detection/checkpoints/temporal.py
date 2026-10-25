"""Separate full-state resume checkpoints and weights-only fine-tuning artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.tasks.ball_detection.model_io.mdd_temporal import (
    TemporalConfig,
    TemporalModel,
    build_temporal_model,
    restore_temporal_model,
)
from src.tasks.ball_detection.models.mdd import (
    MDDHeatmapConfig,
    MDDTemporalQueryConfig,
)
from src.utils.checksum import dual_sha256

RUN_SCHEMA = "ball_temporal_run.v2"
CHECKPOINT_SCHEMA = "ball_temporal_checkpoint.v2"


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


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
    return dict(
        torch=torch.get_rng_state(),
        python=random.getstate(),
        numpy=dict(
            algorithm=numpy[0],
            keys=numpy[1].tolist(),
            position=numpy[2],
            has_gauss=numpy[3],
            cached_gaussian=numpy[4],
        ),
        cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    )


def restore_rng(saved: dict[str, Any], device: torch.device) -> None:
    if (saved["cuda"] is not None) != (device.type == "cuda"):
        raise ValueError("Resume cannot change CPU/CUDA RNG topology")
    torch.set_rng_state(saved["torch"])
    random.setstate(saved["python"])
    numpy = saved["numpy"]
    np.random.set_state(
        (
            numpy["algorithm"],
            np.asarray(numpy["keys"], dtype=np.uint32),
            numpy["position"],
            numpy["has_gauss"],
            numpy["cached_gaussian"],
        )
    )
    if device.type == "cuda":
        if len(saved["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume CUDA device count changed")
        torch.cuda.set_rng_state_all(saved["cuda"])


def log_identity(path: Path) -> dict[str, Any]:
    return dict(bytes=path.stat().st_size, sha256=dual_sha256(path))


def restore_log(path: Path, identity: dict[str, Any], updates: int) -> None:
    with path.open("r+b") as stream:
        prefix = stream.read(identity["bytes"])
        if (
            len(prefix) != identity["bytes"]
            or hashlib.sha256(prefix).hexdigest() != identity["sha256"]
        ):
            raise ValueError("Committed train log changed or is incomplete")
        rows = [json.loads(line) for line in prefix.splitlines()]
        if [row["update"] for row in rows] != list(range(1, updates + 1)):
            raise ValueError("Committed train log is not a continuous update sequence")
        # Updates after the latest checkpoint were never committed; replay those draws.
        stream.truncate(identity["bytes"])


def initialize_finetuning(
    path: Path, *, expected_sha256: str, parent_update: int, config: TemporalConfig
) -> tuple[TemporalModel, dict[str, Any]]:
    if dual_sha256(path) != expected_sha256:
        raise ValueError("Initial heatmap artifact hash mismatch")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    has_optimizer = saved.get("schema") == CHECKPOINT_SCHEMA
    artifact = saved["model_artifact"] if has_optimizer else saved
    previous = restore_temporal_model(
        artifact
    )  # Validate the unmodified saved T first.
    training = artifact.get("training", {})
    if (
        training.get("schema") != RUN_SCHEMA
        or training.get("phase") not in {"pretrain", "finetune"}
        or training.get("update") != parent_update
        or parent_update < 1
    ):
        raise ValueError(
            "Fine-tuning requires the specified update of an explicit temporal training artifact"
        )
    old, new = asdict(previous.config), asdict(config)
    for key in ("frames", "frame_chunk_size", "activation_checkpointing"):
        old.pop(key)
        new.pop(key)
    encoder_only = (
        type(config) is MDDTemporalQueryConfig
        and type(previous.config) is MDDHeatmapConfig
    )
    if encoder_only and (
        previous.config.frames != 32
        or config.frames != 32
        or training["phase"] != "pretrain"
    ):
        raise ValueError(
            "DPT-to-query initialization requires common T32 pretraining and a fresh T32 query"
        )
    if old != new or not encoder_only and type(previous.config) is not type(config):
        raise ValueError(
            "Fine-tuning must preserve architecture and every parameter topology"
        )
    if config.frames < previous.config.frames:
        raise ValueError("Sequential temporal stages cannot decrease T")
    model = build_temporal_model(config, mdd_a=previous.mdd.a, mdd_b=previous.mdd.b)
    weights = {
        k: v
        for k, v in previous.state_dict().items()
        if not encoder_only or k.startswith("encoder.")
    }
    if encoder_only:
        model.encoder.load_state_dict(
            {k.removeprefix("encoder."): v for k, v in weights.items()}, strict=True
        )
    else:
        model.load_state_dict(weights, strict=True)
    for key, value in weights.items():
        other = model.state_dict()[key]
        if value.dtype != other.dtype or not torch.equal(value, other):
            raise ValueError(f"Strict full-state transfer failed: {key}")
    receipt = dict(
        mode="encoder" if encoder_only else "full_temporal_weights",
        artifact_sha256=expected_sha256,
        update=parent_update,
        saved_frames=previous.config.frames,
        target_frames=config.frames,
        training=training,
        transferred_keys=sorted(weights),
        new_keys=sorted(set(model.state_dict()) - weights.keys()),
        has_optimizer_state=has_optimizer,
        sample_position=training["sample_position"],
        cumulative_updates=training["cumulative_updates"],
        cumulative_input_frames=training["cumulative_input_frames"],
    )
    return model, receipt


def optimizer_parameter_names(model: TemporalModel) -> list[list[str]]:
    names = list(dict(model.named_parameters()))
    return [
        [n for n in names if n.startswith("encoder.")],
        [n for n in names if not n.startswith("encoder.")],
    ]


def inherit_optimizer(
    path: Path,
    optimizer: torch.optim.Optimizer,
    model: TemporalModel,
    receipt: dict[str, Any],
) -> None:
    if (
        not receipt["has_optimizer_state"]
        or dual_sha256(path) != receipt["artifact_sha256"]
    ):
        raise ValueError(
            "Optimizer inheritance requires the unchanged full parent checkpoint"
        )
    saved = torch.load(path, map_location="cpu", weights_only=True)
    old = saved["optimizer"]
    states = {
        name: old["state"].get(index)
        for names, group in zip(
            saved["optimizer_parameter_names"], old["param_groups"], strict=True
        )
        for name, index in zip(names, group["params"], strict=True)
    }
    current = optimizer.state_dict()
    transferred = set(receipt["transferred_keys"])
    for names, group in zip(
        optimizer_parameter_names(model), current["param_groups"], strict=True
    ):
        for name, index in zip(names, group["params"], strict=True):
            if name in transferred:
                if name not in states:
                    raise ValueError(
                        f"Parent optimizer parameter identity is missing: {name}"
                    )
                if states[name] is not None:
                    current["state"][index] = copy.deepcopy(states[name])
    optimizer.load_state_dict(current)


def read_resume(
    path: Path, *, output: Path, recipe: dict[str, Any], code: dict[str, Any]
) -> dict[str, Any]:
    latest = json.loads((output / "latest.json").read_text())
    if (
        path.resolve() != (output / latest["checkpoint"]).resolve()
        or dual_sha256(path) != latest["sha256"]
    ):
        raise ValueError(
            "Resume requires this run's latest committed checkpoint and matching hash"
        )
    saved: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(
            "Resume requires an explicit temporal heatmap checkpoint, not a weights artifact"
        )
    if json_bytes(saved["recipe"]) != json_bytes(recipe):
        raise ValueError(
            "Resume recipe changed (including T, budget, data, initialization or runtime)"
        )
    for key in ("source_sha256", "torch_version"):
        if saved["code"][key] != code[key]:
            raise ValueError("Resume code or torch version changed")
    update = saved["updates_completed"]
    budget = recipe["budget"]
    expected_schedule = dict(updates_completed=update, **budget)
    initial_position = (
        recipe["initialization"]["sample_position"] if recipe["initialization"] else 0
    )
    if saved["schedule"] != expected_schedule or saved[
        "sample_position"
    ] != initial_position + update * (budget["input_frames"] // 128):
        raise ValueError("Resume schedule/update/pack position disagree")
    if latest["update"] != update or not 0 < update <= budget["updates"]:
        raise ValueError("Invalid saved update")
    return saved
