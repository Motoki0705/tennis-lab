"""CPU diagnostics on window occurrences with explicit physical sampling rates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np

from src.utils.schema.court import COURT_COORD_SCALE_XYZ


def _stats(values: np.ndarray) -> dict[str, Any]:
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite derived diagnostic values")
    result = {
        "count": int(values.size),
        **{name: float(fn(values)) if values.size else None for name, fn in (
            ("mean", np.mean), ("median", np.median),
            ("p95", lambda x: np.percentile(x, 95)), ("max", np.max),
        )},
    }
    if any(value is not None and not np.isfinite(value) for value in result.values()):
        raise ValueError("Nonfinite diagnostic summary")
    return result


def _variance(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    pred_std = np.std(pred, axis=0) if len(pred) else None
    target_std = np.std(target, axis=0) if len(target) else None
    denominator = float(np.linalg.norm(target_std)) if target_std is not None else 0.0
    ratio = float(np.linalg.norm(pred_std) / denominator) if denominator and pred_std is not None else None
    if (pred_std is not None and not np.isfinite(pred_std).all()) or (target_std is not None and not np.isfinite(target_std).all()) or (ratio is not None and not np.isfinite(ratio)):
        raise ValueError("Nonfinite position variation")
    return {
        "count": len(pred), "std_ddof": 0,
        "pred_std_xyz_m": pred_std.tolist() if pred_std is not None else None,
        "target_std_xyz_m": target_std.tolist() if target_std is not None else None,
        "std_norm_ratio": ratio,
    }


def _validate(arrays: Mapping[str, np.ndarray]) -> tuple[int, int, int]:
    required = ["video_ids", "clip_ids", "camera_ids", "frame_idx", "padding_mask",
                "player_mask", "ball_mask"]
    required += [f"{kind}_{entity}_position" for kind in ("pred", "target") for entity in ("player", "ball")]
    for key in required:
        if key not in arrays:
            raise ValueError(f"Missing evaluation array: {key}")
    if arrays["player_mask"].ndim != 3:
        raise ValueError("player_mask must have shape (windows, players, frames)")
    n, p, t = arrays["player_mask"].shape
    if min(n, p, t) < 1:
        raise ValueError("Evaluation dimensions must be nonempty")
    shapes: dict[str, tuple[int, ...]] = {key: (n,) for key in ("video_ids", "clip_ids", "camera_ids")}
    shapes.update({key: (n, t) for key in ("frame_idx", "padding_mask", "ball_mask")})
    shapes["player_mask"] = (n, p, t)
    for kind in ("pred", "target"):
        shapes[f"{kind}_player_position"] = (n, p, t, 3)
        shapes[f"{kind}_ball_position"] = (n, t, 3)
    for key, shape in shapes.items():
        value = arrays[key]
        if value.shape != shape:
            raise ValueError(f"Invalid shape for {key}: {value.shape}, expected {shape}")
        if key.endswith("_ids"):
            if value.dtype.kind != "U" or any(not item for item in value):
                raise ValueError(f"{key} must contain nonempty string identifiers")
        elif key.endswith("mask"):
            if value.dtype != np.bool_:
                raise ValueError(f"{key} must be boolean")
        elif key == "frame_idx":
            if value.dtype.kind not in "iu" or (value[~arrays["padding_mask"]] < 0).any():
                raise ValueError("frame_idx must contain nonnegative integer physical indices outside padding")
        elif value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError(f"{key} must contain finite real coordinates")
    return n, p, t


def summarize_motion(
    arrays: Mapping[str, np.ndarray],
    fps_by_clip: Mapping[tuple[str, str], float],
    *,
    position_representation: Literal["normalized_court", "meters"],
) -> dict[str, Any]:
    """Summarize positions and forward differences without crossing window boundaries.

    FPS keys are exactly (video_id, full clip_id), with no extra/missing entries.
    Coordinates must explicitly be normalized by COURT_COORD_SCALE_XYZ or meters.
    Derivatives use only adjacent array elements whose physical indices increase
    by one, with all participating target masks true and padding masks false.
    Label weights are not used. All supplied coordinates, including masked ones,
    must be finite. Near/far player slots are summarized separately, not tracked
    identities. Camera groups use (video_id, camera_id) to avoid name collisions.
    """
    n, p, t = _validate(arrays)
    if position_representation not in ("normalized_court", "meters"):
        raise ValueError("position_representation must be normalized_court or meters")
    keys = list(zip(arrays["video_ids"].tolist(), arrays["clip_ids"].tolist(), strict=True))
    if set(fps_by_clip) != set(keys):
        raise ValueError("fps_by_clip must map exactly every evaluated (video_id, full clip_id)")
    for key, value in fps_by_clip.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"FPS must be finite and positive for {key}")
    fps = np.array([fps_by_clip[key] for key in keys], dtype=np.float64)
    scale = np.array(COURT_COORD_SCALE_XYZ) if position_representation == "normalized_court" else np.ones(3)
    positions = {key: np.asarray(value, dtype=np.float64) * scale for key, value in arrays.items()
                 if key in {f"{kind}_{entity}_position" for kind in ("pred", "target") for entity in ("player", "ball")}}
    if any(not np.isfinite(value).all() for value in positions.values()):
        raise ValueError("Nonfinite meter coordinates after conversion")
    entities = [("ball", positions["pred_ball_position"], positions["target_ball_position"], arrays["ball_mask"])]
    entities += [(f"player_slot_{slot}", positions["pred_player_position"][:, slot],
                  positions["target_player_position"][:, slot], arrays["player_mask"][:, slot]) for slot in range(p)]
    # Build derivative vectors once; groups only select their window occurrences.
    samples = {}
    for name, pred, target, mask in entities:
        valid = mask & ~arrays["padding_mask"]
        derivatives = []
        for order in range(1, 4):
            width = max(t - order, 0)
            keep = valid[:, :width].copy()
            for offset in range(1, order + 1):
                keep &= valid[:, offset:offset + width]
                keep &= (arrays["frame_idx"][:, offset:offset + width]
                         == arrays["frame_idx"][:, offset - 1:offset - 1 + width] + 1)
            with np.errstate(over="raise", invalid="raise"):
                pred_diff = np.diff(pred, n=order, axis=1) * fps[:, None, None] ** order
                target_diff = np.diff(target, n=order, axis=1) * fps[:, None, None] ** order
            derivatives.append((pred_diff, target_diff, keep))
        samples[name] = (pred, target, valid, derivatives)
    groups: list[tuple[dict[str, str], np.ndarray]] = [({"group_type": "all"}, np.ones(n, dtype=bool))]
    for video in np.unique(arrays["video_ids"]):
        selection = arrays["video_ids"] == video
        groups.append(({"group_type": "video", "video_id": str(video)}, selection))
        for camera in np.unique(arrays["camera_ids"][selection]):
            groups.append(({"group_type": "camera", "video_id": str(video), "camera_id": str(camera)},
                           selection & (arrays["camera_ids"] == camera)))
    rows = []
    for group, selection in groups:
        results = {}
        for name, (pred, target, valid, derivatives) in samples.items():
            keep = valid & selection[:, None]
            result = {"position_error_m": _stats(np.linalg.norm(pred[keep] - target[keep], axis=-1)),
                      "position_variation": _variance(pred[keep], target[keep])}
            for (label, unit), (pred_diff, target_diff, mask) in zip(
                (("velocity", "m/s"), ("acceleration", "m/s^2"), ("jerk", "m/s^3")), derivatives, strict=True,
            ):
                keep = mask & selection[:, None]
                result[label] = {"unit": unit,
                                 "pred_norm": _stats(np.linalg.norm(pred_diff[keep], axis=-1)),
                                 "target_norm": _stats(np.linalg.norm(target_diff[keep], axis=-1)),
                                 "error_norm": _stats(np.linalg.norm(pred_diff[keep] - target_diff[keep], axis=-1))}
            results[name] = result
        rows.append({**group, "num_windows": int(selection.sum()), "entities": results})
    return {
        "schema_version": 1,
        "aggregation": "Unweighted window occurrences; overlaps counted separately; player slots are near/far ordering, not identities.",
        "interpretation": "Pseudo-teacher agreement; variation ratios do not indicate accuracy or success.",
        "position_representation": position_representation, "position_scale_xyz_m": scale.tolist(),
        "fps_by_clip": [{"video_id": key[0], "clip_id": key[1], "fps": float(fps_by_clip[key])} for key in sorted(set(keys))],
        "rows": rows,
    }
