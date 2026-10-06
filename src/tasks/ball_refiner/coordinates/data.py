"""Load one shared rally store; prepare deterministic corruption in bounded RAM."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.tasks.ball_refiner.coordinates.config import CorruptionConfig
from src.tasks.ball_refiner.coordinates.corruption import (
    CorruptedTrajectory,
    corrupt_trajectory,
)
from src.tasks.ball_refiner.coordinates.generation import SCHEMA


@dataclass(frozen=True)
class Rally:
    index: int
    name: str
    split: str
    xyz: NDArray[np.float32]
    uv: NDArray[np.float32]
    visible: NDArray[np.bool_]
    projection: NDArray[np.float64]
    events: NDArray[np.uint8]
    time: NDArray[np.float64]


class SharedDataset:
    def __init__(self, root: Path) -> None:
        if json.loads((root / "state.json").read_text())["status"] != "complete":
            raise ValueError("The shared dataset is not complete")
        self.manifest_hash = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
        self.manifest: dict[str, Any] = json.loads((root / "manifest.json").read_text())
        if self.manifest["schema"] != SCHEMA:
            raise ValueError("Unsupported shared dataset schema")
        self.image_size = tuple(self.manifest["image_size_wh"])
        if self.image_size != (1280, 720):
            raise ValueError("This noise recipe requires the declared 1280x720 reference grid")
        self.fps = float(self.manifest["fps"])
        self.rallies = []
        identities = set()
        for index, record in enumerate(self.manifest["records"]):
            if record["id"] in identities or record["split"] not in ("train", "val", "test"):
                raise ValueError("Duplicate rally or unknown split")
            identities.add(record["id"])
            path = root / record["path"]
            if not path.resolve().is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Dataset path/hash mismatch: {record['id']}")
            with np.load(path, allow_pickle=False) as payload:
                arrays = {key: payload[key] for key in ("xyz_m", "uv_px", "visible", "projection", "events", "time_s")}
            frames, views = record["frames"], self.manifest["views"]
            shapes = {"xyz_m": (frames, 3), "uv_px": (views, frames, 2), "visible": (views, frames),
                      "projection": (views, 3, 4), "events": (frames,), "time_s": (frames,)}
            if any(arrays[key].shape != shape for key, shape in shapes.items()) or any(not np.isfinite(value).all() for value in arrays.values()):
                raise ValueError(f"Invalid rally arrays: {record['id']}")
            if arrays["visible"].dtype != np.bool_ or arrays["events"].dtype != np.uint8 or (arrays["events"] > 3).any():
                raise ValueError("Visibility/event dtypes are part of the dataset contract")
            if not np.allclose(arrays["time_s"], np.arange(frames) / self.fps):
                raise ValueError("Rally sampling does not match the dataset FPS")
            self.rallies.append(Rally(index, record["id"], record["split"], arrays["xyz_m"], arrays["uv_px"],
                                     arrays["visible"], arrays["projection"], arrays["events"], arrays["time_s"]))
        if {r.split for r in self.rallies} != {"train", "val", "test"}:
            raise ValueError("All three nonempty rally-disjoint splits are required")

    def split(self, split: str) -> list[Rally]:
        if split not in ("train", "val", "test"):
            raise ValueError("Unknown split")
        return [r for r in self.rallies if r.split == split]


def normalization(dimensions: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    if dimensions == 2:
        return np.array([1279, 719], np.float32), np.array([0.5, 0.5], np.float32)
    if dimensions == 3:
        return np.array([10, 20, 5], np.float32), np.zeros(3, np.float32)
    raise ValueError("Require 2D or 3D coordinates")


@dataclass(frozen=True)
class PreparedRally:
    source: Rally
    corrupted: CorruptedTrajectory
    coordinates: NDArray[np.float32]  # V,T,D (3D uses V=1)
    missing: NDArray[np.bool_]
    target: NDArray[np.float32]


def prepare(rallies: list[Rally], dimensions: int, config: CorruptionConfig, seed: int) -> list[PreparedRally]:
    scale, offset = normalization(dimensions)
    output = []
    for rally in rallies:
        corruption_seed = int(np.random.SeedSequence([seed, rally.index]).generate_state(1)[0])
        corrupted = corrupt_trajectory(rally.uv, rally.visible, rally.projection, rally.events, config=config, seed=corruption_seed)
        inputs = corrupted.uv_px if dimensions == 2 else corrupted.xyz_m[None]
        missing = corrupted.missing_2d if dimensions == 2 else corrupted.missing_3d[None]
        truth = rally.uv if dimensions == 2 else rally.xyz[None]
        normalized = np.where(missing[..., None], 0, inputs / scale - offset).astype(np.float32)
        output.append(PreparedRally(rally, corrupted, normalized, missing, (truth / scale - offset).astype(np.float32)))
    return output


def sample_batch(data: list[PreparedRally], batch_size: int, length: int, rng: np.random.Generator, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    coords, masks, targets = [], [], []
    # Equal rally sampling; one random view for 2D. No camera or event features.
    for index in rng.integers(len(data), size=batch_size):
        rally = data[index]
        if rally.coordinates.shape[1] < length:
            raise ValueError("Rally shorter than the configured training window")
        view = int(rng.integers(len(rally.coordinates)))
        start = int(rng.integers(rally.coordinates.shape[1] - length + 1))
        coords.append(rally.coordinates[view, start:start + length])
        masks.append(rally.missing[view, start:start + length])
        targets.append(rally.target[view, start:start + length])
    return tuple(torch.from_numpy(np.stack(values)).to(device) for values in (coords, masks, targets))
