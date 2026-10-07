"""Evaluate full trajectories with shared inference and metrics."""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np
import torch

from src.tasks.ball_refiner_3d.data.schema import PreparedRally
from src.tasks.ball_refiner_3d.evaluation.baselines import linear_baseline
from src.tasks.ball_refiner_3d.evaluation.physics import (
    PhysicsParameters,
    RallyPrediction,
    parameter_report,
    physics_report,
)
from src.tasks.ball_refiner_3d.inference.clip import predict_normalized
from src.tasks.ball_refiner_3d.model_io.adapters import normalization
from src.tasks.ball_refiner_3d.model_io.factory import RefinerModel
from src.tasks.ball_refiner_3d.physics.targets import FlightClock
from src.tasks.ball_refiner_3d.physics.units import decode_field, physical_field
from src.tasks.ball_refiner_3d.training.metrics import error_metrics, event_neighborhood


def evaluate(
    model: RefinerModel,
    data: list[PreparedRally],
    device: torch.device,
    *,
    seed: int,
    batch_size: int,
    physics: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Full-rally predictions and metrics; ``physics`` adds ``physics_eval.v1``.

    The physics protocol fits the force model to every rally, so it runs only
    for saved held-out evaluations, not for per-epoch validation.  Physics heads
    add per-frame arrays ``integrated`` (flights segmented at the predicted
    events, labels in ``integrated_segment``) and ``integrated_truth_segments``
    (ground-truth segmentation, an upper bound), and per-rally arrays
    ``physics_field`` (``physics.units.PHYSICAL_FIELD_COLUMNS``) and
    ``physics_surface_probability`` keyed by ``physics_rally_id``; ``physics``
    adds their reports and the parameter errors.
    """
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
    physics_inputs: list[tuple[RallyPrediction, RallyPrediction]] = []
    integrated: list[np.ndarray] = []
    integrated_segments: list[np.ndarray] = []
    truth_integrated: list[np.ndarray] = []
    fields: list[np.ndarray] = []
    surfaces: list[np.ndarray] = []
    physics_ids: list[int] = []
    integrated_inputs: list[tuple[RallyPrediction, RallyPrediction]] = []
    parameters: list[PhysicsParameters] = []
    heads = model.config.physics_heads
    clock = FlightClock.of([r.source.physics for r in data]) if heads else None
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
            clock=clock,
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
        baseline = linear_baseline(physical_input, rally.missing)
        baselines.append(baseline.reshape(-1, model.config.dimensions))
        upper = None
        if normalized.physics is not None:
            if shape[0] != 1:
                raise ValueError("Physics heads require single-view 3D rallies")
            truth = torch.from_numpy(rally.physics.segment[None]).to(device)
            upper = predict_normalized(
                model,
                coordinates,
                missing,
                batch_size=batch_size,
                seed=seed + rally.source.index,
                clock=clock,
                segment=truth,
            )
            if upper.physics is None:
                raise RuntimeError("Physics heads returned no physics")
            integrated.append(
                (normalized.physics.integrated.cpu().numpy()[0] + offset) * scale
            )
            truth_integrated.append(
                (upper.physics.integrated.cpu().numpy()[0] + offset) * scale
            )
            integrated_segments.append(normalized.physics.segment.cpu().numpy()[0])
            fields.append(
                physical_field(decode_field(upper.physics.field.cpu()))[0].numpy()
            )
            surfaces.append(upper.physics.surface_probability.cpu().numpy()[0])
            physics_ids.append(rally.source.index)
        if physics:
            if shape[0] != 1:
                raise ValueError("Physics evaluation requires single-view 3D rallies")
            probability = normalized.event_probability.cpu().numpy()[0]
            physics_inputs.append(
                (
                    RallyPrediction(
                        rally.source, prediction[0], rally.missing[0], probability
                    ),
                    RallyPrediction(
                        rally.source, baseline[0], rally.missing[0], probability
                    ),
                )
            )
            if upper is not None and upper.physics is not None:
                integrated_inputs.append(
                    (
                        RallyPrediction(
                            rally.source, integrated[-1], rally.missing[0], probability
                        ),
                        RallyPrediction(
                            rally.source,
                            truth_integrated[-1],
                            rally.missing[0],
                            probability,
                        ),
                    )
                )
                parameters.append(
                    PhysicsParameters(
                        rally,
                        upper.physics.field.cpu().numpy()[0],
                        upper.physics.surface_probability.cpu().numpy()[0],
                        upper.physics.segment_states.cpu().numpy()[0],
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
    if heads:
        arrays.update(
            integrated=np.concatenate(integrated),
            integrated_segment=np.concatenate(integrated_segments).astype(np.int64),
            integrated_truth_segments=np.concatenate(truth_integrated),
            physics_rally_id=np.array(physics_ids, dtype=np.int64),
            physics_field=np.stack(fields).astype(np.float32),
            physics_surface_probability=np.stack(surfaces).astype(np.float32),
        )
    report = summarize_predictions(
        arrays,
        elapsed=elapsed,
        input_sha256=digest.hexdigest(),
        method=model.config.architecture,
    )
    if physics:
        report["physics"] = physics_report([p for p, _ in physics_inputs], device)
        report["linear_baseline"]["physics"] = physics_report(
            [b for _, b in physics_inputs], device, events=False
        )
    if physics and heads:
        report["integrated"]["physics"] = physics_report(
            [p for p, _ in integrated_inputs], device, events=False
        )
        upper_bound = [p for _, p in integrated_inputs]
        report["integrated_truth_segments"] = {
            **error_metrics(
                np.concatenate([p.coordinates for p in upper_bound]),
                np.concatenate([p.rally.xyz for p in upper_bound]),
                np.concatenate([p.missing for p in upper_bound]),
                np.concatenate(
                    [event_neighborhood(p.rally.events) for p in upper_bound]
                ),
            ),
            "physics": physics_report(upper_bound, device, events=False),
        }
        report["parameters"] = parameter_report(parameters)
    return report, arrays


def summarize_predictions(
    arrays: dict[str, np.ndarray], *, elapsed: float, input_sha256: str, method: str
) -> dict[str, Any]:
    report: dict[str, Any] = error_metrics(
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
    if "integrated" in arrays:
        report["integrated"] = error_metrics(
            arrays["integrated"], arrays["target"], arrays["missing"], arrays["event"]
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
