"""All-frame, gap, observed and event metrics on identical fixed corruption."""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.ball_refiner.coordinates.data import PreparedRally, normalization
from src.tasks.ball_refiner.coordinates.inference import (
    RefinerModel,
    predict_normalized,
)


def event_neighborhood(events: np.ndarray, radius: int = 5) -> NDArray[np.bool_]:
    selected: NDArray[np.bool_] = np.zeros(events.shape, dtype=bool)
    for index in np.flatnonzero(events):
        selected[max(0, index - radius):min(len(events), index + radius + 1)] = True
    return selected


def error_metrics(prediction: np.ndarray, target: np.ndarray, missing: np.ndarray, events: np.ndarray) -> dict[str, Any]:
    distance = np.linalg.norm(prediction - target, axis=-1)
    if not np.isfinite(distance).all():
        raise ValueError("Nonfinite evaluation error")
    result: dict[str, Any] = {"frame_missing_rate": float(missing.mean()), "frames": int(missing.size)}
    for name, mask in (("all", np.ones_like(missing)), ("missing", missing), ("observed", ~missing), ("event", events)):
        values = distance[mask]
        result[name] = {"count": len(values), "rmse": float(np.sqrt(np.mean(values**2))) if len(values) else None,
                        "mean": float(values.mean()) if len(values) else None,
                        "p95": float(np.quantile(values, 0.95)) if len(values) else None}
    return result


def linear_baseline(coordinates: np.ndarray, missing: np.ndarray) -> np.ndarray:
    result = coordinates.copy()
    for view in range(len(result)):
        known = np.flatnonzero(~missing[view])
        # Explicit all-missing baseline is zero in physical coordinates.
        for axis in range(result.shape[-1]):
            result[view, :, axis] = np.interp(np.arange(result.shape[1]), known, coordinates[view, known, axis]) if len(known) else 0
    return result


def evaluate(model: RefinerModel, data: list[PreparedRally], device: torch.device, *, seed: int, batch_size: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    predictions: list[np.ndarray] = []
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
        normalized = predict_normalized(model, coordinates, missing, batch_size=batch_size, seed=seed + rally.source.index)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - start
        prediction = (normalized.cpu().numpy() + offset) * scale
        target = (rally.target + offset) * scale
        physical_input = np.where(rally.missing[..., None], 0, (rally.coordinates + offset) * scale)
        events = np.broadcast_to(event_neighborhood(rally.source.events), rally.missing.shape)
        shape = rally.missing.shape
        predictions.append(prediction.reshape(-1, model.config.dimensions))
        targets.append(target.reshape(-1, model.config.dimensions))
        masks.append(rally.missing.ravel())
        event_masks.append(events.ravel())
        inputs.append(physical_input.reshape(-1, model.config.dimensions))
        baselines.append(linear_baseline(physical_input, rally.missing).reshape(-1, model.config.dimensions))
        ids.append(np.full(np.prod(shape), rally.source.index, dtype=np.int64))
        view_ids.append(np.repeat(np.arange(shape[0]), shape[1]))
        frame_ids.append(np.tile(np.arange(shape[1]), shape[0]))
        digest.update(rally.source.name.encode())
        digest.update(rally.coordinates.tobytes())
        digest.update(rally.missing.tobytes())
    arrays = {name: np.concatenate(values) for name, values in (
        ("prediction", predictions), ("target", targets), ("missing", masks), ("event", event_masks), ("input", inputs),
        ("linear_baseline", baselines), ("rally_id", ids), ("view_id", view_ids), ("frame_id", frame_ids),
    )}
    report = error_metrics(arrays["prediction"], arrays["target"], arrays["missing"], arrays["event"])
    report["linear_baseline"] = error_metrics(arrays["linear_baseline"], arrays["target"], arrays["missing"], arrays["event"])
    report.update({"unit": "px" if model.config.dimensions == 2 else "m", "inference_seconds": elapsed,
                   "milliseconds_per_frame": elapsed * 1000 / report["frames"], "evaluation_input_sha256": digest.hexdigest(),
                   "sampling": "one fixed-seed flow trajectory; no oracle selection or sample averaging"})
    return report, arrays
