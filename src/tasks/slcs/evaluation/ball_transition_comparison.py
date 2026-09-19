"""Paired ball velocity diagnostics on saved, unchanged evaluation windows."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.tasks.slcs.data.augmentation import INPUT_CONDITIONS
from src.tasks.slcs.evaluation.comparison import MATCH_KEYS
from src.utils.schema.court import COURT_COORD_SCALE_XYZ

_PAIRED_KEYS = (*MATCH_KEYS, "ball_observed", "player_observed")


def _validate(arrays: dict[str, np.ndarray]) -> None:
    if arrays["player_mask"].ndim != 3:
        raise ValueError("player_mask must have shape (windows, players, frames)")
    n, p, t = arrays["player_mask"].shape
    if min(n, p, t) < 1:
        raise ValueError("Evaluation dimensions must be nonempty")
    shapes: dict[str, tuple[int, ...]] = {key: (n,) for key in MATCH_KEYS[:6]}
    shapes.update(
        {
            key: (n, t)
            for key in (
                "frame_idx",
                "padding_mask",
                "ball_mask",
                "ball_weight",
                "ball_observed",
            )
        }
    )
    shapes.update(
        {key: (n, p, t) for key in ("player_mask", "player_weight", "player_observed")}
    )
    shapes.update(
        target_player_position=(n, p, t, 3),
        target_player_rotation=(n, p, t, 2),
        target_ball_position=(n, t, 3),
        pred_ball_position=(n, t, 3),
    )
    for key, shape in shapes.items():
        value = arrays[key]
        if value.shape != shape:
            raise ValueError(f"Invalid shape for {key}: expected {shape}")
        if key.endswith("_ids"):
            if value.dtype.kind != "U" or any(not item for item in value):
                raise ValueError(f"Invalid identifiers: {key}")
        elif key.endswith(("mask", "observed")):
            if value.dtype != np.bool_:
                raise ValueError(f"{key} must be boolean")
        elif key in {"frame_idx", "window_start", "window_length"}:
            if value.dtype.kind not in "iu":
                raise ValueError(f"{key} must contain integers")
        elif value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError(f"{key} must contain finite real values")
    if len(set(arrays["scene_ids"].tolist())) != n:
        raise ValueError("scene_ids must be unique")
    padding = arrays["padding_mask"]
    length = arrays["window_length"]
    if (arrays["window_start"] < 0).any() or ((length < 1) | (length > t)).any():
        raise ValueError("Invalid window start/length")
    if not np.array_equal(padding, np.arange(t)[None] >= length[:, None]):
        raise ValueError("padding_mask disagrees with window_length")
    if (arrays["frame_idx"][~padding] < 0).any() or not np.array_equal(
        arrays["frame_idx"][:, 0], arrays["window_start"]
    ):
        raise ValueError("Invalid physical frame indices/window start")
    for entity in ("ball", "player"):
        mask, weight = arrays[f"{entity}_mask"], arrays[f"{entity}_weight"]
        entity_padding = padding if entity == "ball" else padding[:, None]
        if (
            (mask & entity_padding).any()
            or ((weight < 0) | (weight > 1)).any()
            or (weight[~mask] != 0).any()
        ):
            raise ValueError(f"Invalid {entity} target mask/confidence")


def _load(
    directory: Path,
) -> tuple[dict[str, np.ndarray], dict[tuple[str, str], float], str, str]:
    with np.load(directory / "eval_arrays.npz", allow_pickle=False) as archive:
        required = (*_PAIRED_KEYS, "pred_ball_position")
        if set(required) - set(archive.files):
            raise ValueError(
                f"Missing evaluation arrays: {sorted(set(required) - set(archive.files))}"
            )
        arrays = {key: archive[key] for key in required}
    _validate(arrays)
    motion = json.loads((directory / "motion.json").read_text())
    if not isinstance(motion, dict):
        raise ValueError("motion.json must be a mapping")
    if motion.get("position_representation") != "normalized_court" or motion.get(
        "position_scale_xyz_m"
    ) != list(COURT_COORD_SCALE_XYZ):
        raise ValueError("Expected normalized_court and exact COURT_COORD_SCALE_XYZ")
    entries = motion.get("fps_by_clip")
    if not isinstance(entries, list):
        raise ValueError("fps_by_clip must be an explicit list")
    fps: dict[tuple[str, str], float] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"video_id", "clip_id", "fps"}:
            raise ValueError("Malformed fps_by_clip entry")
        if any(
            not isinstance(entry[key], str) or not entry[key]
            for key in ("video_id", "clip_id")
        ):
            raise ValueError("Invalid FPS identifiers")
        key = (entry["video_id"], entry["clip_id"])
        value = entry["fps"]
        if (
            key in fps
            or type(value) not in (int, float)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("FPS must be unique, finite and positive")
        fps[key] = float(value)
    keys = set(
        zip(arrays["video_ids"].tolist(), arrays["clip_ids"].tolist(), strict=True)
    )
    if set(fps) != keys:
        raise ValueError(
            "fps_by_clip must cover exactly all evaluated (video_id, clip_id) keys"
        )
    config = OmegaConf.load(directory / "evaluation_config.yaml")
    if not isinstance(config, DictConfig):
        raise ValueError("evaluation_config must be a mapping")
    mode, split = (
        OmegaConf.select(config, "evaluate.input_mode"),
        OmegaConf.select(config, "evaluate.split"),
    )
    if (
        not isinstance(mode, str)
        or not isinstance(split, str)
        or mode not in INPUT_CONDITIONS
        or split not in {"train", "val", "test"}
    ):
        raise ValueError("Explicit valid input condition and split are required")
    return arrays, fps, str(mode), str(split)


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
    base, fps, mode, split = _load(baseline)
    other, other_fps, other_mode, other_split = _load(candidate)
    if (fps, mode, split) != (other_fps, other_mode, other_split):
        raise ValueError("FPS, input condition or split mismatch")
    for key in _PAIRED_KEYS:
        if base[key].dtype != other[key].dtype or not np.array_equal(
            base[key], other[key]
        ):
            raise ValueError(f"Unmatched paired array: {key}")
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
