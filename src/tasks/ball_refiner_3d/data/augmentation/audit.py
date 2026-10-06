"""Report realized corruption independently of optimization."""

from __future__ import annotations

import numpy as np

from src.tasks.ball_refiner_3d.data.schema import PreparedRally


def corruption_audit(data: list[PreparedRally]) -> dict[str, float]:
    noises = np.concatenate(
        [
            np.linalg.norm(r.corrupted.noise_px, axis=-1)[~r.corrupted.missing_2d]
            for r in data
        ]
    )
    eligible = sum(
        np.count_nonzero(
            r.source.visible
            & ~np.any(
                r.corrupted.missing_2d & ~r.corrupted.isolated & r.source.visible,
                axis=0,
            )[None]
        )
        for r in data
    )
    event_count = sum(np.count_nonzero(r.source.events) for r in data)
    return {
        "noise_p95_px": float(np.quantile(noises, 0.95)),
        "noise_p50_px": float(np.median(noises)),
        "frame_missing_rate_2d": float(
            np.concatenate([r.corrupted.missing_2d.ravel() for r in data]).mean()
        ),
        "frame_missing_rate_3d": float(
            np.concatenate([r.corrupted.missing_3d for r in data]).mean()
        ),
        "isolated_probability_realized": sum(
            int(r.corrupted.isolated.sum()) for r in data
        )
        / eligible
        if eligible
        else 0.0,
        "selected_event_fraction": sum(len(r.corrupted.intervals) for r in data)
        / event_count
        if event_count
        else 0.0,
    }
