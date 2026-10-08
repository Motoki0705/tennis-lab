"""Physical-coordinate drawing and diagnostic payloads for one shared rally."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.tasks.ball_refiner_3d.data.schema import CorruptedTrajectory, Rally
from src.tasks.ball_refiner_3d.data.targets.events import gaussian_event_target
from src.tasks.ball_refiner_3d.evaluation.baselines import linear_baseline
from src.tasks.ball_refiner_3d.evaluation.physics import DEFAULT_PROTOCOL
from src.tasks.ball_refiner_3d.physics.units import PHYSICAL_FIELD_COLUMNS, SURFACES
from src.tasks.ball_refiner_3d.training.metrics import error_metrics, event_neighborhood
from src.tasks.base.visualization.review.court import court_edges, court_keypoints
from src.utils.physics.ball.kinematics import KinematicThresholds, kinematic_report


@dataclass(frozen=True)
class PhysicsOutputs:
    """Physics-head outputs of one rally in physical units."""

    integrated: np.ndarray  # (T,3) m, flights segmented at predicted events
    segment: np.ndarray  # (T,) predicted flight labels
    integrated_truth_segments: np.ndarray  # (T,3) m, ground-truth flights
    field: np.ndarray  # (4,) PHYSICAL_FIELD_COLUMNS
    surface_probability: np.ndarray  # (3,) SURFACES

    def __post_init__(self) -> None:
        frames = len(self.segment)
        if (
            self.integrated.shape != (frames, 3)
            or self.integrated_truth_segments.shape != (frames, 3)
            or self.field.shape != (len(PHYSICAL_FIELD_COLUMNS),)
            or self.surface_probability.shape != (len(SURFACES),)
        ):
            raise ValueError("物理headの出力の形が不正です")
        if not all(
            np.isfinite(value).all()
            for value in (
                self.integrated,
                self.integrated_truth_segments,
                self.field,
                self.surface_probability,
            )
        ):
            raise ValueError("物理headの出力に非有限値があります")
        if frames and (
            self.segment[0] != 0 or np.any(~np.isin(np.diff(self.segment), (0, 1)))
        ):
            raise ValueError("予測区間のラベルが0から連続していません")


@dataclass(frozen=True)
class ModelOutputs:
    """One checkpoint's outputs for one rally; ``physics`` only for physics heads."""

    coordinates: np.ndarray  # (T,3) m
    event_probability: np.ndarray  # (T,)
    physics: PhysicsOutputs | None


def metrics(
    prediction: np.ndarray, rally: Rally, missing: np.ndarray
) -> dict[str, Any]:
    """Coordinate errors and in-flight kinematics of ``physics_eval.v1``."""
    result: dict[str, Any] = error_metrics(
        prediction, rally.xyz, missing, event_neighborhood(rally.events)
    )
    result["kinematics"] = kinematic_report(
        prediction.astype(np.float64),
        rally.xyz.astype(np.float64),
        rally.physics.frame_segment(),
        float(rally.physics.output_fps),
        {"all": np.ones(len(missing), dtype=bool), "missing": missing},
        KinematicThresholds(
            DEFAULT_PROTOCOL.acceleration_mps2, DEFAULT_PROTOCOL.below_ground_m
        ),
    )
    return result


def scene_payload(
    rally: Rally,
    corruption: CorruptedTrajectory,
    cameras: dict[str, np.ndarray],
    *,
    fps: float,
    model: ModelOutputs | None,
    event_sigma_frames: float,
) -> dict[str, Any]:
    points = court_keypoints(None)
    event_mask: np.ndarray = np.zeros(len(rally.events), dtype=bool)
    for _, start, end in corruption.intervals:
        event_mask[start:end] = True
    observed = ~corruption.missing_2d
    noise = np.linalg.norm(corruption.noise_px[observed], axis=-1)
    missing = corruption.missing_3d
    linear = linear_baseline(
        np.where(missing[:, None], 0, corruption.xyz_m)[None], missing[None]
    )[0]
    physics = model.physics if model is not None else None
    series = {
        "linear": linear,
        "prediction": model.coordinates if model is not None else None,
        "integrated": physics.integrated if physics is not None else None,
        "integrated_truth": physics.integrated_truth_segments
        if physics is not None
        else None,
    }
    record = rally.physics
    return {
        "rally": rally.name,
        "split": rally.split,
        "frames": len(rally.xyz),
        "fps": fps,
        "time": rally.time.tolist(),
        "events": rally.events.tolist(),
        "intervals": corruption.intervals.tolist(),
        "event_missing": event_mask.tolist(),
        "missing_3d": missing.tolist(),
        "gt_3d": rally.xyz.tolist(),
        "input_3d": corruption.xyz_m.tolist(),
        **{
            f"{kind}_3d": None if values is None else values.tolist()
            for kind, values in series.items()
        },
        "metrics": {
            kind: metrics(values, rally, missing)
            for kind, values in series.items()
            if values is not None
        },
        "segments": {
            "truth": record.frame_segment().tolist(),
            "predicted": physics.segment.tolist() if physics is not None else None,
        },
        "physics": {
            "columns": list(PHYSICAL_FIELD_COLUMNS),
            "truth": [*map(float, record.wind[:2]), record.k_drag, record.k_magnus],
            "predicted": physics.field.tolist() if physics is not None else None,
            "surfaces": list(SURFACES),
            "surface": record.surface,
            "surface_probability": physics.surface_probability.tolist()
            if physics is not None
            else None,
        },
        "court": {"keypoints": points.tolist(), "edges": court_edges()},
        "cameras": [
            {
                "id": f"cam_{v}",
                "label": f"Camera {v + 1}",
                "params": {
                    "C": cameras["camera_centers"][v].tolist(),
                    "R": cameras["rotation"][v].tolist(),
                    "f": float(cameras["intrinsic"][v, 0, 0]),
                    "cx": float(cameras["intrinsic"][v, 0, 2]),
                    "cy": float(cameras["intrinsic"][v, 1, 2]),
                    "w": 1280,
                    "h": 720,
                },
            }
            for v in range(len(rally.uv))
        ],
        "event_probability": model.event_probability.tolist()
        if model is not None
        else None,
        "event_target": gaussian_event_target(
            rally.events, event_sigma_frames
        ).tolist(),
        "event_sigma_frames": event_sigma_frames,
        "audit": {
            "frame_missing_rate_3d": float(missing.mean()),
            "selected_events": len(corruption.intervals),
            "events": int(np.count_nonzero(rally.events)),
            "noise_p95_px": float(np.quantile(noise, 0.95)) if len(noise) else None,
        },
    }
