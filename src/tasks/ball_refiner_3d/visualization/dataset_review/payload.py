"""Physical-coordinate drawing and diagnostic payloads for one shared rally."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.tasks.ball_refiner_3d.data.schema import CorruptedTrajectory, Rally
from src.tasks.ball_refiner_3d.data.targets.events import gaussian_event_target
from src.tasks.ball_refiner_3d.training.metrics import error_metrics, event_neighborhood
from src.tasks.base.visualization.review.court import court_edges, court_keypoints


def metrics(
    prediction: np.ndarray | None,
    target: np.ndarray,
    missing: np.ndarray,
    events: np.ndarray,
) -> dict[str, Any] | None:
    if prediction is None:
        return None
    result: dict[str, Any] = error_metrics(
        prediction,
        target,
        missing,
        np.broadcast_to(event_neighborhood(events), missing.shape),
    )
    result["velocity_rmse"] = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (np.diff(prediction, axis=-2) - np.diff(target, axis=-2)) ** 2,
                    axis=-1,
                )
            )
        )
    )
    return result


def scene_payload(
    rally: Rally,
    corruption: CorruptedTrajectory,
    cameras: dict[str, np.ndarray],
    *,
    fps: float,
    prediction_3d: np.ndarray | None,
    event_probability: np.ndarray | None,
    event_sigma_frames: float,
) -> dict[str, Any]:
    points = court_keypoints(None)
    event_mask: np.ndarray = np.zeros(len(rally.events), dtype=bool)
    for _, start, end in corruption.intervals:
        event_mask[start:end] = True
    observed = ~corruption.missing_2d
    noise = np.linalg.norm(corruption.noise_px[observed], axis=-1)
    three_metrics = metrics(
        prediction_3d, rally.xyz, corruption.missing_3d, rally.events
    )
    if three_metrics is not None:
        three_metrics["velocity_rmse"] *= fps
    return {
        "rally": rally.name,
        "split": rally.split,
        "frames": len(rally.xyz),
        "fps": fps,
        "time": rally.time.tolist(),
        "events": rally.events.tolist(),
        "intervals": corruption.intervals.tolist(),
        "event_missing": event_mask.tolist(),
        "missing_3d": corruption.missing_3d.tolist(),
        "gt_3d": rally.xyz.tolist(),
        "input_3d": corruption.xyz_m.tolist(),
        "prediction_3d": prediction_3d.tolist() if prediction_3d is not None else None,
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
        "metrics_3d": three_metrics,
        "event_probability": event_probability.tolist()
        if event_probability is not None
        else None,
        "event_target": gaussian_event_target(
            rally.events, event_sigma_frames
        ).tolist(),
        "event_sigma_frames": event_sigma_frames,
        "audit": {
            "frame_missing_rate_3d": float(corruption.missing_3d.mean()),
            "selected_events": len(corruption.intervals),
            "events": int(np.count_nonzero(rally.events)),
            "noise_p95_px": float(np.quantile(noise, 0.95)) if len(noise) else None,
        },
    }
