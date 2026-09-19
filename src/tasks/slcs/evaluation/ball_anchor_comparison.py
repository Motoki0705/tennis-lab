"""Saved validation diagnostics by window-local observed ball anchor availability."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.slcs.evaluation.paired_ball_arrays import load_paired_ball_evaluations
from src.utils.schema.court import COURT_COORD_SCALE_XYZ

ANCHOR_CLASSES = (
    "observed",
    "missing_left_only",
    "missing_right_only",
    "missing_both_sides",
    "missing_no_anchor",
)
# Fixed before evaluation: burst augmentation reaches 24 frames. Never fit bins
# to the baseline/candidate predictions or evaluation targets.
_DISTANCE_BUCKETS = (("1", 1, 1), ("2-8", 2, 8), ("9-24", 9, 24), (">=25", 25, None))


def _anchor_layout(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    padding = arrays["padding_mask"]
    frames = arrays["frame_idx"]
    adjacent = ~padding[:, :-1] & ~padding[:, 1:]
    if np.any(adjacent & (np.diff(frames, axis=1) != 1)):
        raise ValueError(
            "Anchor diagnostics require contiguous real frames in each window"
        )
    observed = arrays["ball_observed"] & ~padding
    length = observed.shape[1]
    index = np.broadcast_to(np.arange(length), observed.shape)
    left = np.maximum.accumulate(np.where(observed, index, -1), axis=1)
    right = np.minimum.accumulate(np.where(observed, index, length)[:, ::-1], axis=1)[
        :, ::-1
    ]
    has_left, has_right = left >= 0, right < length
    missing = ~observed & ~padding
    labels = np.full(observed.shape, "padding", dtype="U24")
    labels[observed] = "observed"
    labels[missing & has_left & ~has_right] = "missing_left_only"
    labels[missing & ~has_left & has_right] = "missing_right_only"
    labels[missing & has_left & has_right] = "missing_both_sides"
    labels[missing & ~has_left & ~has_right] = "missing_no_anchor"
    # None for observed/no-anchor/padding. Under the checked contiguous contract,
    # window offset differences equal physical frame_idx differences exactly.
    distances = np.full(observed.shape, None, dtype=object)
    nearest = np.minimum(
        np.where(has_left, index - left, length),
        np.where(has_right, right - index, length),
    )
    defined = missing & (has_left | has_right)
    distances[defined] = nearest[defined]
    return labels, distances


def _buckets(
    labels: np.ndarray, distances: np.ndarray, anchor_class: str
) -> list[tuple[str, np.ndarray]]:
    selected = labels == anchor_class
    buckets = [("all", selected)]
    if anchor_class in {"observed", "missing_no_anchor"}:
        return buckets
    flat_distances: list[int | None] = distances.ravel().tolist()
    for name, minimum, maximum in _DISTANCE_BUCKETS:
        mask = np.fromiter(
            (
                value is not None
                and value >= minimum
                and (maximum is None or value <= maximum)
                for value in flat_distances
            ),
            dtype=bool,
            count=distances.size,
        ).reshape(distances.shape)
        buckets.append((name, selected & mask))
    return buckets


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite derived anchor diagnostic")
    result: dict[str, float | int | None] = {"count": int(values.size)}
    for name, fn in (
        ("mean", np.mean),
        ("p95", lambda x: np.percentile(x, 95)),
        ("max", np.max),
    ):
        result[name] = float(fn(values)) if values.size else None
    if any(value is not None and not np.isfinite(value) for value in result.values()):
        raise ValueError("Nonfinite anchor summary")
    return result


def _paired_stats(base: np.ndarray, other: np.ndarray) -> dict[str, Any]:
    baseline, candidate = _stats(base), _stats(other)
    delta: dict[str, float | int | None] = {}
    for key, base_value in baseline.items():
        candidate_value = candidate[key]
        delta[key] = (
            None
            if base_value is None or candidate_value is None
            else candidate_value - base_value
        )
    return {
        "baseline": baseline,
        "candidate": candidate,
        "candidate_minus_baseline": delta,
    }


def _source(directory: Path) -> dict[str, Any]:
    return {
        "directory": str(directory.resolve()),
        "sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("eval_arrays.npz", "motion.json", "evaluation_config.yaml")
        },
    }


def compare_ball_anchors(baseline: Path, candidate: Path) -> dict[str, Any]:
    """Compare unchanged paired validation arrays; never evaluate a test split."""
    base, other, fps, mode, split = load_paired_ball_evaluations(baseline, candidate)
    if split != "val":
        raise ValueError("Anchor comparison requires the validation split (val)")
    labels, distances = _anchor_layout(base)
    scale = np.array(COURT_COORD_SCALE_XYZ, dtype=np.float64)
    rates = np.array(
        [
            fps[(video, clip)]
            for video, clip in zip(base["video_ids"], base["clip_ids"], strict=True)
        ]
    )
    with np.errstate(over="raise", invalid="raise"):
        target = base["target_ball_position"].astype(np.float64) * scale
        predictions = [
            arrays["pred_ball_position"].astype(np.float64) * scale
            for arrays in (base, other)
        ]
        errors = [np.linalg.norm(pred - target, axis=-1) for pred in predictions]
        teacher_velocity = np.diff(target, axis=1) * rates[:, None, None]
        teacher_speed = np.linalg.norm(teacher_velocity, axis=-1)
        velocities = [
            np.diff(pred, axis=1) * rates[:, None, None] for pred in predictions
        ]
        speeds = [np.linalg.norm(velocity, axis=-1) for velocity in velocities]
        velocity_errors = [
            np.linalg.norm(velocity - teacher_velocity, axis=-1)
            for velocity in velocities
        ]
    valid = base["ball_mask"] & ~base["padding_mask"]
    pairs = valid[:, :-1] & valid[:, 1:] & (np.diff(base["frame_idx"], axis=1) == 1)
    observed = base["ball_observed"] & ~base["padding_mask"]
    videos = base["video_ids"]
    groups: list[tuple[str, str, np.ndarray]] = [
        ("all", "all", np.ones(len(videos), dtype=bool))
    ]
    groups += [("video", str(video), videos == video) for video in np.unique(videos)]
    position_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    strata = [
        (anchor_class, bucket, mask)
        for anchor_class in ANCHOR_CLASSES
        for bucket, mask in _buckets(labels, distances, anchor_class)
    ]
    for group_type, group, windows in groups:
        group_info = {
            "group_type": group_type,
            "group": group,
            "num_windows": int(windows.sum()),
        }
        for anchor_class, bucket, mask in strata:
            keep = valid & windows[:, None] & mask
            position_rows.append(
                {
                    **group_info,
                    "anchor_class": anchor_class,
                    "distance_bucket": bucket,
                    **_paired_stats(errors[0][keep], errors[1][keep]),
                }
            )
            if anchor_class == "observed":
                continue
            for transition, boundary, endpoint in (
                (
                    "observed_to_missing",
                    observed[:, :-1] & ~observed[:, 1:],
                    mask[:, 1:],
                ),
                (
                    "missing_to_observed",
                    ~observed[:, :-1] & observed[:, 1:],
                    mask[:, :-1],
                ),
            ):
                pair_keep = pairs & windows[:, None] & boundary & endpoint
                metrics = {
                    "velocity_vector_error_mps": velocity_errors,
                    "pred_speed_mps": speeds,
                    "teacher_speed_mps": [teacher_speed, teacher_speed],
                    "signed_speed_bias_mps": [
                        speed - teacher_speed for speed in speeds
                    ],
                }
                boundary_rows.append(
                    {
                        **group_info,
                        "transition": transition,
                        "anchor_class": anchor_class,
                        "distance_bucket": bucket,
                        "metrics": {
                            name: _paired_stats(
                                values[0][pair_keep], values[1][pair_keep]
                            )
                            for name, values in metrics.items()
                        },
                    }
                )
    return {
        "schema_version": 1,
        "sources": {"baseline": _source(baseline), "candidate": _source(candidate)},
        "input_condition": mode,
        "split": split,
        "position_error_unit": "m",
        "position_representation": "normalized_court",
        "position_scale_xyz_m": scale.tolist(),
        "fps_by_clip": [
            {"video_id": video, "clip_id": clip, "fps": fps[(video, clip)]}
            for video, clip in sorted(fps)
        ],
        "anchor_definition": "Original ball_observed & ~padding_mask only, independently inside each window. Missing frames use nearest left/right observed anchors; both-sides distance is the nearer anchor. Observed/no-anchor distance is None. Noncontiguous real frames are rejected.",
        "distance_buckets": ["all", *[name for name, _, _ in _DISTANCE_BUCKETS]],
        "distance_bucket_basis": "Fixed before evaluation using burst augmentation maximum 24 and prior train-only observation distribution; no fitting to evaluation data. all overlaps its distance subsets; observed/no-anchor have all only.",
        "boundary_distance_definition": "Distance belongs to the missing endpoint, never the observed endpoint. Every retained observed/missing boundary has a directly adjacent observed anchor, so missing-endpoint distance is exactly 1; no-anchor and larger-distance boundary strata are empty by construction.",
        "aggregation": "Unweighted valid ball teacher frame/pair occurrences; overlapping windows counted separately. Position requires a valid nonpadding teacher frame; velocity requires both endpoints valid, nonpadding and frame_idx difference 1. Confidence is matched but never weights statistics. Empty strata have count 0 and null statistics. p95 uses linear interpolation.",
        "interpretation": "Pseudo-teacher agreement, not measured 3D accuracy. Deltas are candidate minus baseline. Signed speed bias is predicted speed magnitude minus teacher speed magnitude, not vector error. No clipping, filtering or smoothing is applied.",
        "position_rows": position_rows,
        "boundary_rows": boundary_rows,
    }


def save_ball_anchor_comparison(
    baseline: Path, candidate: Path, *, output: Path
) -> Path:
    """Save one new absolute JSON artifact without replacing files or symlinks."""
    if not output.is_absolute() or output.suffix != ".json":
        raise ValueError("output must be an absolute JSON path")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing existing output: {output}")
    report = compare_ball_anchors(baseline, candidate)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(payload)
    return output
