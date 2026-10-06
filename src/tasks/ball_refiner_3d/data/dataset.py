"""Read and validate the immutable shared rally store."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.ball_refiner_3d.data.schema import SCHEMA, Rally


class SharedDataset:
    def __init__(self, root: Path) -> None:
        if json.loads((root / "state.json").read_text())["status"] != "complete":
            raise ValueError("The shared dataset is not complete")
        self.manifest_hash = hashlib.sha256(
            (root / "manifest.json").read_bytes()
        ).hexdigest()
        self.manifest: dict[str, Any] = json.loads((root / "manifest.json").read_text())
        if self.manifest["schema"] != SCHEMA:
            raise ValueError("Unsupported shared dataset schema")
        self.image_size = tuple(self.manifest["image_size_wh"])
        if self.image_size != (1280, 720):
            raise ValueError(
                "This noise recipe requires the declared 1280x720 reference grid"
            )
        self.fps = float(self.manifest["fps"])
        self.rallies = []
        identities = set()
        for index, record in enumerate(self.manifest["records"]):
            if record["id"] in identities or record["split"] not in (
                "train",
                "val",
                "test",
            ):
                raise ValueError("Duplicate rally or unknown split")
            identities.add(record["id"])
            path = root / record["path"]
            if (
                not path.resolve().is_relative_to(root.resolve())
                or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]
            ):
                raise ValueError(f"Dataset path/hash mismatch: {record['id']}")
            with np.load(path, allow_pickle=False) as payload:
                arrays = {
                    key: payload[key]
                    for key in (
                        "xyz_m",
                        "uv_px",
                        "visible",
                        "projection",
                        "events",
                        "time_s",
                    )
                }
            frames, views = record["frames"], self.manifest["views"]
            shapes = {
                "xyz_m": (frames, 3),
                "uv_px": (views, frames, 2),
                "visible": (views, frames),
                "projection": (views, 3, 4),
                "events": (frames,),
                "time_s": (frames,),
            }
            if any(arrays[key].shape != shape for key, shape in shapes.items()) or any(
                not np.isfinite(value).all() for value in arrays.values()
            ):
                raise ValueError(f"Invalid rally arrays: {record['id']}")
            if (
                arrays["visible"].dtype != np.bool_
                or arrays["events"].dtype != np.uint8
                or (arrays["events"] > 3).any()
            ):
                raise ValueError(
                    "Visibility/event dtypes are part of the dataset contract"
                )
            if not np.allclose(arrays["time_s"], np.arange(frames) / self.fps):
                raise ValueError("Rally sampling does not match the dataset FPS")
            self.rallies.append(
                Rally(
                    index,
                    record["id"],
                    record["split"],
                    arrays["xyz_m"],
                    arrays["uv_px"],
                    arrays["visible"],
                    arrays["projection"],
                    arrays["events"],
                    arrays["time_s"],
                )
            )
        if {r.split for r in self.rallies} != {"train", "val", "test"}:
            raise ValueError("All three nonempty rally-disjoint splits are required")

    def split(self, split: str) -> list[Rally]:
        if split not in ("train", "val", "test"):
            raise ValueError("Unknown split")
        return [r for r in self.rallies if r.split == split]
