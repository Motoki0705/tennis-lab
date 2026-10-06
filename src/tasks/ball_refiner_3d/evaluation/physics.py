"""Physical-consistency evaluation of predicted rallies (protocol ``physics_eval.v1``).

The protocol is fixed in code so every run and baseline is measured identically:

- kinematics: in-flight finite-difference velocity/acceleration errors, jerk
  ratio, accelerations above ``acceleration_mps2`` (ground truth peaks at
  32 m/s^2 over the regenerated 1,280 rallies) and frames below ground;
- fit: the residual of the shared force model fitted to the prediction over the
  ground-truth flight segments (gravity fixed);
- events: probability peaks matched to ground-truth event frames;
- breakdowns by surface, rally length and bounce count.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.ball_refiner_3d.data.schema import Rally
from src.tasks.ball_refiner_3d.training.metrics import event_neighborhood
from src.utils.physics.ball.events import event_detection_report, pick_event_frames
from src.utils.physics.ball.fitting import FitSettings, fit_flight_physics
from src.utils.physics.ball.kinematics import KinematicThresholds, kinematic_report
from src.utils.physics.ball.record import EVENT_BITS, EVENT_KINDS

PROTOCOL = "physics_eval.v1"


@dataclass(frozen=True)
class PhysicsProtocol:
    acceleration_mps2: float = 40.0
    below_ground_m: float = 0.02
    fit_iterations: int = 20
    minimum_segment_frames: int = 3
    event_threshold: float = 0.5
    event_min_separation: int = 3
    event_tolerance: int = 2
    length_bins: tuple[int, ...] = (256, 512)
    bounce_bins: tuple[int, ...] = (3, 4, 5)


DEFAULT_PROTOCOL = PhysicsProtocol()


@dataclass(frozen=True)
class RallyPrediction:
    """Physical-unit prediction of one rally, single view."""

    rally: Rally
    coordinates: NDArray[np.floating]  # (T,3) m
    missing: NDArray[np.bool_]  # (T,)
    event_probability: NDArray[np.floating]  # (T,)


def _rmse(values: NDArray[np.floating]) -> float | None:
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else None


def _bins(value: int, edges: tuple[int, ...]) -> str:
    lower = 0
    for edge in edges:
        if value < edge:
            return f"[{lower},{edge})"
        lower = edge
    return f"[{lower},inf)"


def physics_report(
    predictions: list[RallyPrediction],
    device: torch.device,
    protocol: PhysicsProtocol = DEFAULT_PROTOCOL,
    *,
    events: bool = True,
) -> dict[str, Any]:
    """Physics metrics pooled over rallies; ``events`` skips peak matching."""
    if not predictions:
        raise ValueError("Require at least one rally")
    fps = predictions[0].rally.physics.output_fps
    first = predictions[0].rally.physics
    if any(p.rally.physics.output_fps != fps for p in predictions):
        raise ValueError("All rallies must share one output FPS")
    fits = fit_flight_physics(
        [p.coordinates.astype(np.float64) for p in predictions],
        [
            [(s.start_frame, s.end_frame) for s in p.rally.physics.segments]
            for p in predictions
        ],
        FitSettings(
            gravity=first.gravity,
            dt=first.dt,
            substeps=first.stride,
            iterations=protocol.fit_iterations,
            minimum_segment_frames=protocol.minimum_segment_frames,
        ),
        device,
    )
    pooled: dict[str, list[NDArray[np.floating]]] = {}
    kinematic_inputs = []
    groups: dict[str, dict[str, list[NDArray[np.float64]]]] = {
        "surface": {},
        "frames": {},
        "bounces": {},
    }
    for prediction, fit in zip(predictions, fits, strict=True):
        record = prediction.rally.physics
        segment = record.frame_segment()
        opening = np.array(
            [record.event_kind[s.event] for s in record.segments], dtype=np.int64
        )[segment]
        masks = _frame_masks(prediction, opening)
        kinematic_inputs.append((prediction, segment, masks))
        residual = np.linalg.norm(fit.residual, axis=-1)
        error = np.linalg.norm(prediction.coordinates - prediction.rally.xyz, axis=-1)
        for name, mask in masks.items():
            selected = mask & fit.fitted
            pooled.setdefault(f"fit_{name}", []).append(residual[selected])
        bounces = int(np.count_nonzero(prediction.rally.events & EVENT_BITS["bounce"]))
        tags = {
            "surface": record.surface,
            "frames": _bins(record.frames, protocol.length_bins),
            "bounces": _bins(bounces, protocol.bounce_bins),
        }
        for kind, tag in tags.items():
            entry = groups[kind].setdefault(tag, [])
            entry.append(np.stack((error, np.where(fit.fitted, residual, np.nan))))
    report: dict[str, Any] = {
        "protocol": PROTOCOL,
        "settings": asdict(protocol),
        "fit_residual_rmse": {
            name.removeprefix("fit_"): _rmse(np.concatenate(values))
            for name, values in pooled.items()
        },
        "kinematics": _pooled_kinematics(kinematic_inputs, fps, protocol),
        "breakdown": {
            kind: {
                tag: _breakdown(np.concatenate(values, axis=1))
                for tag, values in sorted(entries.items())
            }
            for kind, entries in groups.items()
        },
    }
    if events:
        report["events"] = event_detection_report(
            [
                pick_event_frames(
                    p.event_probability,
                    protocol.event_threshold,
                    protocol.event_min_separation,
                )
                for p in predictions
            ],
            [np.flatnonzero(p.rally.events) for p in predictions],
            protocol.event_tolerance,
        )
    return report


def _frame_masks(
    prediction: RallyPrediction, opening: NDArray[np.int64]
) -> dict[str, NDArray[np.bool_]]:
    frames = len(prediction.missing)
    return {
        "all": np.ones(frames, dtype=bool),
        "missing": prediction.missing,
        "observed": ~prediction.missing,
        "event": event_neighborhood(prediction.rally.events),
        "after_bounce": opening == EVENT_KINDS.index("bounce"),
        "after_shot": opening == EVENT_KINDS.index("shot"),
    }


def _pooled_kinematics(
    inputs: list[
        tuple[RallyPrediction, NDArray[np.int64], dict[str, NDArray[np.bool_]]]
    ],
    fps: float,
    protocol: PhysicsProtocol,
) -> dict[str, Any]:
    """Concatenate rallies with a separator label so no stencil crosses them."""
    positions: list[NDArray[np.floating]] = []
    targets: list[NDArray[np.floating]] = []
    segments: list[NDArray[np.int64]] = []
    masks: dict[str, list[NDArray[np.bool_]]] = {}
    offset = 0
    for prediction, segment, frame_masks in inputs:
        positions.append(prediction.coordinates)
        targets.append(prediction.rally.xyz)
        segments.append(np.append(segment + offset, -1))
        offset += int(segment.max()) + 1
        for name, mask in frame_masks.items():
            masks.setdefault(name, []).append(np.append(mask, False))
        positions.append(np.zeros((1, 3)))
        targets.append(np.zeros((1, 3)))
    return kinematic_report(
        np.concatenate(positions).astype(np.float64),
        np.concatenate(targets).astype(np.float64),
        np.concatenate(segments),
        fps,
        {name: np.concatenate(values) for name, values in masks.items()},
        KinematicThresholds(protocol.acceleration_mps2, protocol.below_ground_m),
    )


def _breakdown(values: NDArray[np.float64]) -> dict[str, Any]:
    error, residual = values
    finite = residual[np.isfinite(residual)]
    return {
        "frames": int(error.size),
        "rmse": _rmse(error),
        "fit_residual_rmse": _rmse(finite),
    }


def physics_metrics(report: dict[str, Any]) -> dict[str, float]:
    """Scalar headline metrics of a :func:`physics_report`.

    Event precision is left to the full report: it is undefined when a model
    fires no peak, while F1 and recall are defined whenever events exist.
    """
    kinematics = report["kinematics"]["all"]
    events = report["events"]
    values = {
        "test_fit_residual_rmse_m": report["fit_residual_rmse"]["all"],
        "test_fit_residual_missing_rmse_m": report["fit_residual_rmse"]["missing"],
        "test_velocity_rmse_mps": kinematics["velocity_rmse"],
        "test_acceleration_rmse_mps2": kinematics["acceleration_rmse"],
        "test_jerk_ratio": kinematics["jerk_ratio"],
        "test_implausible_acceleration_rate": kinematics[
            "implausible_acceleration_rate"
        ],
        "test_event_f1": events["f1"],
        "test_event_recall": events["recall"],
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ValueError(f"Physics metrics undefined: {missing}")
    return {name: float(value) for name, value in values.items()}
