"""Atomic serialization of one generated rally and its provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.ball_refiner_3d.configuration.generation import (
    CameraSampling,
    GenerationConfig,
)
from src.tasks.blcs.generate_dataset.simulation.rally_simulator import RallyResult
from src.utils.geometry.triangulation import PinholeCamera
from src.utils.io import write_json_atomic
from src.utils.physics.ball.record import BallPhysicsRecord


def write_rally(
    root: Path,
    index: int,
    generation: GenerationConfig,
    camera: CameraSampling,
    split: str,
    result: RallyResult,
    record: BallPhysicsRecord,
    xyz: np.ndarray,
    cameras: tuple[PinholeCamera, ...],
    attempts: int,
) -> dict[str, Any]:
    name = f"rally_{index:06d}"
    seed = generation.seed + index
    if record.frames != len(xyz):
        raise ValueError(f"{name}: physics record and trajectory lengths differ")
    points, visible = [], []
    for view in cameras:
        uv, front = view.project(xyz)
        if not front.all():
            # Keep only projection-finite trajectories: never fabricate GT behind a camera.
            raise ValueError(f"{name}: trajectory passed behind a rear-fence camera")
        points.append(uv.astype(np.float32))
        visible.append(
            front
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < camera.width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < camera.height)
        )
    arrays = {
        "xyz_m": xyz,
        "uv_px": np.stack(points),
        "visible": np.stack(visible),
        "time_s": np.arange(len(xyz), dtype=np.float64) / record.output_fps,
        "projection": np.stack([view.matrix for view in cameras]),
        "camera_centers": np.stack([view.center for view in cameras]),
        "intrinsic": np.stack([view.intrinsic for view in cameras]),
        "rotation": np.stack([view.rotation for view in cameras]),
        "translation": np.stack([view.translation for view in cameras]),
        **record.to_arrays(),
    }
    if not np.isfinite(xyz).all():
        raise ValueError(f"Nonfinite physical trajectory: {name}")
    path = root / "rallies" / f"{name}.npz"
    with path.with_suffix(".partial").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    path.with_suffix(".partial").replace(path)
    metadata = {
        "id": name,
        "split": split,
        "seed": seed,
        "path": f"rallies/{name}.npz",
        "frames": len(xyz),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "attempts": attempts,
        "shots": len(result.shot_events),
        "surface": record.surface,
        "segments": len(record.segments),
        "event_frames": int(np.count_nonzero(record.event_mask())),
        "end_reason": result.end_reason.value,
        "visible_fraction": float(np.mean(visible)),
        "shot_metadata_fps": result.fps_out,
        "shot_metadata": [
            {
                key: getattr(shot, key)
                for key in (
                    "shot_index",
                    "t_start",
                    "t_return",
                    "t_bounce1",
                    "t_bounce2",
                    "t_bounce3",
                    "shot_type",
                    "return_type",
                )
            }
            for shot in result.shot_events
        ],
    }
    write_json_atomic(path.with_suffix(".json"), metadata)
    return metadata
