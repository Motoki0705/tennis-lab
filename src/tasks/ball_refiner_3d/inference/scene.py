"""Event inference on the original frame grid of a SceneResult v2 clip."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from src.tasks.ball_refiner_3d.inference.predictor import RefinerPredictor
from src.tasks.ball_refiner_3d.inference.segmentation import (
    DEFAULT_PICKING,
    EventPicking,
    pick_class_event_frames,
)
from src.tasks.ball_refiner_3d.model_io.contracts import (
    RefinerPrediction,
    validate_input,
)
from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.schema import SceneResult


def scene_input(scene: SceneResult) -> tuple[Tensor, Tensor, float]:
    """Extract (1,T,3) court metres, (1,T) true=missing, and the actual clip FPS."""
    if scene.metadata.get("scene_schema_version") != 2:
        raise ValueError("Refiner scene input requires SceneResult v2")
    ball, valid = scene.ball_3d, scene.ball_3d_valid
    if (
        ball is None
        or ball.shape != (scene.num_frames, 3)
        or valid is None
        or valid.dtype != np.bool_
        or valid.shape != (scene.num_frames,)
        or not valid.any()
    ):
        raise ValueError(
            "Require aligned ball_3d and ball_3d_valid with observed ball positions"
        )
    if not math.isfinite(scene.fps) or scene.fps <= 0:
        raise ValueError("Scene FPS must be positive and finite")
    coordinates = torch.from_numpy(
        np.where(valid[:, None], ball, 0).astype(np.float32)[None]
    )
    missing = torch.from_numpy(~valid[None])
    validate_input(coordinates, missing, 3)
    return coordinates, missing, scene.fps


def predict_scene_events(
    predictor: RefinerPredictor,
    scene_path: Path,
    *,
    clip_id: str,
    checkpoint_sha256: str,
    seed: int,
    fps_tolerance: float = 0.0,
    picking: EventPicking = DEFAULT_PICKING,
) -> tuple[RefinerPrediction, dict[str, Any]]:
    """Produce per-frame probabilities and the shared sparse prediction document."""
    scene_sha256 = hashlib.sha256(scene_path.read_bytes()).hexdigest()
    scene = load_scene_result(scene_path)
    coordinates, missing, fps = scene_input(scene)
    prediction = predictor.predict(
        coordinates, missing, fps=fps, seed=seed, fps_tolerance=fps_tolerance
    )
    probability = prediction.event_probability[0].numpy()
    events = [
        {
            "kind": kind,
            "frame": int(frame),
            "time_s": float(frame / fps),
            "score": float(probability[frame, channel]),
            "features": {},
            "flags": ["ball_3d_missing"] if missing[0, frame] else [],
        }
        for channel, (kind, frames) in enumerate(
            pick_class_event_frames(probability, picking).items()
        )
        for frame in frames
    ]
    return prediction, {
        "schema": "ball_event_predictions.v1",
        "clip_id": clip_id,
        "fps": fps,
        "frames": scene.num_frames,
        "method": "refiner3",
        "method_version": checkpoint_sha256,
        "inputs": {"scene_sha256": scene_sha256},
        "status": "ok",
        "reason": None,
        "events": sorted(events, key=lambda event: (event["frame"], event["kind"])),
    }
