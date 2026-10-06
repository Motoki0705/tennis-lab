"""Physical-coordinate drawing and diagnostic payloads for one shared rally."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.tasks.ball_refiner.coordinates.corruption import CorruptedTrajectory
from src.tasks.ball_refiner.coordinates.data import Rally
from src.tasks.ball_refiner.coordinates.evaluation import (
    error_metrics,
    event_neighborhood,
)
from src.tasks.base.visualization.review.court import court_edges, court_keypoints


def metrics(prediction: np.ndarray | None, target: np.ndarray, missing: np.ndarray, events: np.ndarray) -> dict[str, Any] | None:
    if prediction is None:
        return None
    result: dict[str, Any] = error_metrics(prediction, target, missing, np.broadcast_to(event_neighborhood(events), missing.shape))
    result["velocity_rmse"] = float(np.sqrt(np.mean(np.sum((np.diff(prediction, axis=-2) - np.diff(target, axis=-2)) ** 2, axis=-1))))
    return result


def scene_payload(rally: Rally, corruption: CorruptedTrajectory, cameras: dict[str, np.ndarray], *, fps: float,
                  prediction_2d: np.ndarray | None, prediction_3d: np.ndarray | None) -> dict[str, Any]:
    points = court_keypoints(None)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    projected = np.einsum("vij,pj->vpi", rally.projection, homogeneous)
    uv_court = projected[..., :2] / projected[..., 2:]
    court_views = [[xy.tolist() if depth > 0 else None for xy, depth in zip(view, depths, strict=True)]
                   for view, depths in zip(uv_court, projected[..., 2], strict=True)]
    event_mask: np.ndarray = np.zeros(len(rally.events), dtype=bool)
    for _, start, end in corruption.intervals:
        event_mask[start:end] = True
    observed = ~corruption.missing_2d
    noise = np.linalg.norm(corruption.noise_px[observed], axis=-1)
    two_metrics = [metrics(prediction_2d[v] if prediction_2d is not None else None, rally.uv[v], corruption.missing_2d[v], rally.events)
                   for v in range(len(rally.uv))]
    three_metrics = metrics(prediction_3d, rally.xyz, corruption.missing_3d, rally.events)
    for result in [*two_metrics, three_metrics]:
        if result is not None:
            result["velocity_rmse"] *= fps
    return {
        "rally": rally.name, "split": rally.split, "frames": len(rally.xyz), "fps": fps, "image_size": [1280, 720],
        "time": rally.time.tolist(), "events": rally.events.tolist(), "intervals": corruption.intervals.tolist(),
        "event_missing": event_mask.tolist(), "isolated_missing": corruption.isolated.tolist(),
        "missing_2d": corruption.missing_2d.tolist(), "missing_3d": corruption.missing_3d.tolist(),
        "gt_2d": rally.uv.tolist(), "input_2d": corruption.uv_px.tolist(),
        "prediction_2d": prediction_2d.tolist() if prediction_2d is not None else None,
        "gt_3d": rally.xyz.tolist(), "input_3d": corruption.xyz_m.tolist(),
        "prediction_3d": prediction_3d.tolist() if prediction_3d is not None else None,
        "court": {"keypoints": points.tolist(), "edges": court_edges()}, "court_2d": court_views,
        "cameras": [{"id": f"cam_{v}", "label": f"Camera {v + 1}", "params": {
            "C": cameras["camera_centers"][v].tolist(), "R": cameras["rotation"][v].tolist(),
            "f": float(cameras["intrinsic"][v, 0, 0]), "cx": float(cameras["intrinsic"][v, 0, 2]),
            "cy": float(cameras["intrinsic"][v, 1, 2]), "w": 1280, "h": 720}}
            for v in range(len(rally.uv))],
        "metrics_2d": two_metrics, "metrics_3d": three_metrics,
        "audit": {"frame_missing_rate_2d": float(corruption.missing_2d.mean()),
                  "frame_missing_rate_3d": float(corruption.missing_3d.mean()),
                  "selected_events": len(corruption.intervals), "events": int(np.count_nonzero(rally.events)),
                  "noise_p95_px": float(np.quantile(noise, 0.95)) if len(noise) else None,
                  "observed_frames_2d": int(observed.sum())},
    }
