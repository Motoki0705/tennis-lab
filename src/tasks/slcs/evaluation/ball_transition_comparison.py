"""Paired ball velocity diagnostics on saved, unchanged evaluation windows."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.slcs.evaluation.paired_ball_arrays import load_paired_ball_evaluations
from src.utils.schema.court import COURT_COORD_SCALE_XYZ


def _summary(pred: np.ndarray, teacher: np.ndarray) -> dict[str, float | int | None]:
    pred_speed, teacher_speed = (
        np.linalg.norm(pred, axis=-1),
        np.linalg.norm(teacher, axis=-1),
    )
    values = {
        "pred_speed": pred_speed,
        "teacher_speed": teacher_speed,
        "velocity_vector_error": np.linalg.norm(pred - teacher, axis=-1),
    }
    result: dict[str, float | int | None] = {"count": len(pred)}
    for name, samples in values.items():
        if not np.isfinite(samples).all():
            raise ValueError("Nonfinite derived velocity diagnostics")
        for stat, fn in (
            ("mean", np.mean),
            ("p95", lambda x: np.percentile(x, 95)),
            ("max", np.max),
        ):
            result[f"{name}_{stat}_mps"] = float(fn(samples)) if len(samples) else None
    result["signed_speed_bias_mean_mps"] = (
        float(np.mean(pred_speed - teacher_speed)) if len(pred) else None
    )
    if any(value is not None and not math.isfinite(value) for value in result.values()):
        raise ValueError("Nonfinite velocity summary")
    return result


def compare_ball_transitions(
    baseline: Path, candidate: Path, *, fast_speed_mps: float
) -> dict[str, Any]:
    """Compare matched vector errors, not merely smaller predicted speeds."""
    if (
        isinstance(fast_speed_mps, bool)
        or not math.isfinite(fast_speed_mps)
        or fast_speed_mps <= 0
    ):
        raise ValueError("fast_speed_mps must be an explicit finite positive threshold")
    base, other, fps, mode, split = load_paired_ball_evaluations(baseline, candidate)
    rates = np.array(
        [
            fps[(video, clip)]
            for video, clip in zip(base["video_ids"], base["clip_ids"], strict=True)
        ]
    )
    scale = np.array(COURT_COORD_SCALE_XYZ, dtype=np.float64)
    with np.errstate(over="raise", invalid="raise"):
        # Same float64 conversion then forward difference as summarize_motion.
        teacher = (
            np.diff(base["target_ball_position"].astype(np.float64) * scale, axis=1)
            * rates[:, None, None]
        )
        predictions = [
            np.diff(arrays["pred_ball_position"].astype(np.float64) * scale, axis=1)
            * rates[:, None, None]
            for arrays in (base, other)
        ]
    valid = base["ball_mask"] & ~base["padding_mask"]
    pair = valid[:, :-1] & valid[:, 1:] & (np.diff(base["frame_idx"], axis=1) == 1)
    left, right = base["ball_observed"][:, :-1], base["ball_observed"][:, 1:]
    strata = {
        "all": np.ones_like(pair),
        "observed_to_missing": left & ~right,
        "missing_to_observed": ~left & right,
        "both_observed": left & right,
        "both_missing": ~left & ~right,
    }
    fast = np.linalg.norm(teacher, axis=-1) >= fast_speed_mps
    videos = base["video_ids"]
    groups: list[tuple[str, str, np.ndarray]] = [
        ("all", "all", np.ones(len(videos), dtype=bool))
    ]
    groups += [("video", str(video), videos == video) for video in np.unique(videos)]
    rows = []
    for group_type, group, windows in groups:
        for stratum, visibility in strata.items():
            for subset, speed_mask in (
                ("all", np.ones_like(pair)),
                ("fast_teacher", fast),
            ):
                keep = pair & windows[:, None] & visibility & speed_mask
                scores = [_summary(pred[keep], teacher[keep]) for pred in predictions]
                delta: dict[str, float | int | None] = {}
                for key, base_value in scores[0].items():
                    candidate_value = scores[1][key]
                    delta[key] = (
                        None
                        if base_value is None or candidate_value is None
                        else candidate_value - base_value
                    )
                rows.append(
                    {
                        "group_type": group_type,
                        "group": group,
                        "num_windows": int(windows.sum()),
                        "stratum": stratum,
                        "subset": subset,
                        "baseline": scores[0],
                        "candidate": scores[1],
                        "candidate_minus_baseline": delta,
                    }
                )
    return {
        "schema_version": 1,
        "sources": {
            "baseline": str(baseline.resolve()),
            "candidate": str(candidate.resolve()),
        },
        "input_condition": mode,
        "split": split,
        "fast_speed_mps": fast_speed_mps,
        "fast_definition": "teacher speed >= explicit threshold; never fitted on these evaluation arrays",
        "unit": "m/s",
        "position_representation": "normalized_court",
        "position_scale_xyz_m": scale.tolist(),
        "fps_by_clip": [
            {"video_id": key[0], "clip_id": key[1], "fps": fps[key]}
            for key in sorted(fps)
        ],
        "aggregation": "Unweighted adjacent valid ball pairs within windows; both endpoints nonpadding and frame_idx difference 1. Overlapping window occurrences counted separately; confidence is matched but does not weight statistics. Stratum all overlaps four disjoint visibility strata; fast_teacher is a subset of each all-speed stratum.",
        "interpretation": "Pseudo-teacher agreement only. Smaller predicted speeds alone do not indicate success. Deltas are candidate minus baseline; negative vector error favors candidate, signed speed bias can have either sign.",
        "rows": rows,
    }


def save_ball_transition_comparison(
    baseline: Path, candidate: Path, *, fast_speed_mps: float, output: Path
) -> Path:
    """Write one new absolute JSON artifact; never replace an existing file."""
    if not output.is_absolute() or output.suffix != ".json":
        raise ValueError("output must be an absolute JSON path")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing existing output: {output}")
    report = compare_ball_transitions(
        baseline, candidate, fast_speed_mps=fast_speed_mps
    )
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(payload)
    return output
