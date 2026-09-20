"""Paired 3D diagnostics that expose median failures and tail-dominated gains."""

from __future__ import annotations

from typing import Any

import numpy as np


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    if not values.size:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
    }


def paired_errors(
    initial: np.ndarray,
    predicted: np.ndarray,
    target: np.ndarray,
    frame_valid: np.ndarray,
) -> dict[str, Any]:
    """Compare identical valid N,T,J points; padded frames never enter metrics.

    Improvements have a 0.1 mm deadband. The large-correction contribution may
    exceed one when all remaining points collectively get worse; do not clip it.
    """
    if (
        initial.shape != predicted.shape
        or initial.shape != target.shape
        or initial.ndim != 4
        or initial.shape[-1] != 3
        or frame_valid.shape != initial.shape[:2]
        or frame_valid.dtype != np.bool_
    ):
        raise ValueError("Expected paired N,T,J,3 arrays and boolean N,T validity")
    mask = np.broadcast_to(frame_valid[..., None], initial.shape[:-1])
    for value in (initial, predicted, target):
        if not np.isfinite(value[mask]).all():
            raise ValueError("Nonfinite evaluated 3D points must not be hidden")
    initial, predicted, target = (
        np.where(mask[..., None], value, 0) for value in (initial, predicted, target)
    )
    before = np.linalg.norm(initial - target, axis=-1)
    after = np.linalg.norm(predicted - target, axis=-1)
    correction = np.linalg.norm(predicted - initial, axis=-1)
    b, a, c = before[mask], after[mask], correction[mask]
    gain = b - a
    result: dict[str, Any] = {
        "initial_m": _distribution(b),
        "predicted_m": _distribution(a),
        "correction_m": _distribution(c),
        "gain_m": _distribution(gain),
        "improved_fraction": float(np.mean(gain > 1e-4)) if gain.size else None,
        "worsened_fraction": float(np.mean(gain < -1e-4)) if gain.size else None,
    }
    sample_count = mask.sum(axis=(1, 2))
    have_samples = sample_count > 0
    sample_before = np.where(mask, before, 0).sum(axis=(1, 2)) / np.maximum(
        sample_count, 1
    )
    sample_after = np.where(mask, after, 0).sum(axis=(1, 2)) / np.maximum(
        sample_count, 1
    )
    result["sample_mean_initial_m"] = _distribution(sample_before[have_samples])
    result["sample_mean_predicted_m"] = _distribution(sample_after[have_samples])
    result["sample_improved_fraction"] = (
        float(np.mean((sample_before - sample_after)[have_samples] > 1e-4))
        if have_samples.any()
        else None
    )
    if gain.size:
        count = max(1, int(np.ceil(0.01 * gain.size)))
        largest = np.argsort(c, kind="stable")[-count:]
        total_gain = float(gain.sum())
        result["largest_correction_one_percent"] = {
            "point_count": count,
            "net_gain_share": float(gain[largest].sum() / total_gain)
            if abs(total_gain) > 1e-10
            else None,
            "remaining_mean_gain_m": float(
                (gain.sum() - gain[largest].sum()) / (gain.size - count)
            )
            if gain.size > count
            else None,
        }
    return result


def prediction_diagnostics(payload: dict[str, np.ndarray], task: str) -> dict[str, Any]:
    """Summarize world/root/relative errors and recorded sample strata."""
    if task not in ("plcs", "blcs"):
        raise ValueError("Unknown residual task")
    initial, predicted, target = (
        np.asarray(payload[k], dtype=np.float64)
        for k in ("initial_world", "pred_world", "target_world")
    )
    valid = payload["frame_valid"]
    result: dict[str, Any] = {"world": paired_errors(initial, predicted, target, valid)}
    initial, predicted, target = (
        np.where(valid[..., None, None], x, 0) for x in (initial, predicted, target)
    )
    indices = (11, 12) if task == "plcs" else (0,)
    roots = [
        x[:, :, indices].mean(axis=2, keepdims=True)
        for x in (initial, predicted, target)
    ]
    result["root"] = paired_errors(roots[0], roots[1], roots[2], valid)
    if task == "plcs":
        relative = [
            x - root
            for x, root in zip((initial, predicted, target), roots, strict=True)
        ]
        result["relative"] = paired_errors(relative[0], relative[1], relative[2], valid)
    groups: dict[str, np.ndarray] = {}
    if "severity" in payload:
        severity = payload["severity"].reshape(-1)
        groups.update(clean=severity == 0, normal=severity == 1, hard=severity > 1)
    for key in ("corruption_family", "num_views"):
        if key in payload:
            values = payload[key].reshape(-1)
            groups.update(
                {f"{key}_{int(value)}": values == value for value in np.unique(values)}
            )
    result["strata"] = {
        name: paired_errors(
            initial[selected], predicted[selected], target[selected], valid[selected]
        )
        for name, selected in groups.items()
    }
    error = np.linalg.norm(initial - target, axis=-1)
    result["initial_error_bins"] = {}
    for lower, upper in (
        (0.0, 0.05),
        (0.05, 0.2),
        (0.2, 0.5),
        (0.5, 1.0),
        (1.0, np.inf),
    ):
        point_mask = valid[..., None] & (error >= lower) & (error < upper)
        # Treat individual points as samples for this joint-level conditioning.
        points = [
            x[point_mask].reshape(-1, 1, 1, 3) for x in (initial, predicted, target)
        ]
        point_valid: np.ndarray = np.ones((len(points[0]), 1), dtype=bool)
        result["initial_error_bins"][f"{lower:g}_{upper:g}_m"] = paired_errors(
            points[0], points[1], points[2], point_valid
        )
    if "persistent_mask" in payload:
        event_mask: np.ndarray = payload["persistent_mask"].astype(bool)
        if event_mask.shape != initial.shape[:-1]:
            raise ValueError("Persistent provenance must be N,T,J")
        result["persistent"] = {}
        for name, selected in (("event", event_mask), ("outside_event", ~event_mask)):
            point_mask = selected & valid[..., None]
            points = [
                x[point_mask].reshape(-1, 1, 1, 3) for x in (initial, predicted, target)
            ]
            result["persistent"][name] = paired_errors(
                points[0], points[1], points[2], np.ones((len(points[0]), 1), bool)
            )
    if task == "plcs":
        result["wrists"] = paired_errors(
            initial[:, :, [9, 10]],
            predicted[:, :, [9, 10]],
            target[:, :, [9, 10]],
            valid,
        )
    if "init_valid" in payload and "severity" in payload:
        clean_valid = (
            valid[..., None]
            & payload["init_valid"]
            & (payload["severity"].reshape(-1, 1, 1) == 0)
        )
        result["clean_raw_valid_correction_m"] = _distribution(
            np.linalg.norm(predicted - initial, axis=-1)[clean_valid]
        )
    result["input_audit"] = {}
    for key in ("geometry_attempts", "calibration_attempts", "persistent_fraction"):
        if key in payload:
            result["input_audit"][key] = _distribution(payload[key].reshape(-1))
    if "calibration_failed_candidates" in payload:
        result["input_audit"]["calibration_failed_candidates"] = int(
            payload["calibration_failed_candidates"].sum()
        )
    return result
