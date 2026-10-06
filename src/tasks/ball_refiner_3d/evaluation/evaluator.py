"""Evaluate full trajectories with shared inference and metrics."""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np
import torch

from src.tasks.ball_refiner_3d.data.schema import PreparedRally
from src.tasks.ball_refiner_3d.evaluation.baselines import linear_baseline
from src.tasks.ball_refiner_3d.inference.windowing import predict_normalized
from src.tasks.ball_refiner_3d.model_io.adapters import normalization
from src.tasks.ball_refiner_3d.model_io.factory import RefinerModel
from src.tasks.ball_refiner_3d.training.metrics import error_metrics, event_neighborhood


def evaluate(
    model: RefinerModel,
    data: list[PreparedRally],
    device: torch.device,
    *,
    seed: int,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    predictions: list[np.ndarray] = []
    event_probabilities: list[np.ndarray] = []
    event_targets: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    event_masks: list[np.ndarray] = []
    inputs: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    view_ids: list[np.ndarray] = []
    frame_ids: list[np.ndarray] = []
    baselines: list[np.ndarray] = []
    scale, offset = normalization(model.config.dimensions)
    elapsed = 0.0
    digest = hashlib.sha256()
    for rally in data:
        coordinates = torch.from_numpy(rally.coordinates).to(device)
        missing = torch.from_numpy(rally.missing).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        normalized = predict_normalized(
            model,
            coordinates,
            missing,
            batch_size=batch_size,
            seed=seed + rally.source.index,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - start
        prediction = (normalized.coordinates.cpu().numpy() + offset) * scale
        target = (rally.target + offset) * scale
        physical_input = np.where(
            rally.missing[..., None], 0, (rally.coordinates + offset) * scale
        )
        events = np.broadcast_to(
            event_neighborhood(rally.source.events), rally.missing.shape
        )
        shape = rally.missing.shape
        event_probabilities.append(normalized.event_probability.cpu().numpy().ravel())
        event_targets.append(rally.event_target.ravel())
        predictions.append(prediction.reshape(-1, model.config.dimensions))
        targets.append(target.reshape(-1, model.config.dimensions))
        masks.append(rally.missing.ravel())
        event_masks.append(events.ravel())
        inputs.append(physical_input.reshape(-1, model.config.dimensions))
        baselines.append(
            linear_baseline(physical_input, rally.missing).reshape(
                -1, model.config.dimensions
            )
        )
        ids.append(np.full(np.prod(shape), rally.source.index, dtype=np.int64))
        view_ids.append(np.repeat(np.arange(shape[0]), shape[1]))
        frame_ids.append(np.tile(np.arange(shape[1]), shape[0]))
        digest.update(rally.source.name.encode())
        digest.update(rally.coordinates.tobytes())
        digest.update(rally.missing.tobytes())
    arrays = {
        name: np.concatenate(values)
        for name, values in (
            ("event_probability", event_probabilities),
            ("event_target", event_targets),
            ("prediction", predictions),
            ("target", targets),
            ("missing", masks),
            ("event", event_masks),
            ("input", inputs),
            ("linear_baseline", baselines),
            ("rally_id", ids),
            ("view_id", view_ids),
            ("frame_id", frame_ids),
        )
    }
    report = summarize_predictions(
        arrays,
        elapsed=elapsed,
        input_sha256=digest.hexdigest(),
        method=model.config.architecture,
    )
    return report, arrays


def summarize_predictions(
    arrays: dict[str, np.ndarray], *, elapsed: float, input_sha256: str, method: str
) -> dict[str, Any]:
    report = error_metrics(
        arrays["prediction"], arrays["target"], arrays["missing"], arrays["event"]
    )
    report["event_probability"] = {
        "brier": float(
            np.mean((arrays["event_probability"] - arrays["event_target"]) ** 2)
        ),
        "soft_cross_entropy": float(
            np.mean(
                -arrays["event_target"]
                * np.log(np.clip(arrays["event_probability"], 1e-7, 1 - 1e-7))
                - (1 - arrays["event_target"])
                * np.log(np.clip(1 - arrays["event_probability"], 1e-7, 1 - 1e-7))
            )
        ),
    }
    report["linear_baseline"] = error_metrics(
        arrays["linear_baseline"], arrays["target"], arrays["missing"], arrays["event"]
    )
    report.update(
        {
            "unit": "m",
            "inference_seconds": elapsed,
            "milliseconds_per_frame": elapsed * 1000 / report["frames"],
            "evaluation_input_sha256": input_sha256,
            "sampling": "one fixed-seed flow trajectory; no oracle selection or sample averaging"
            if method == "flow"
            else "one deterministic regression trajectory",
        }
    )
    return report
