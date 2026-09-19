"""Strict paired saved-array contract shared by ball diagnostics."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.tasks.slcs.data.augmentation import INPUT_CONDITIONS
from src.tasks.slcs.evaluation.comparison import MATCH_KEYS
from src.utils.schema.court import COURT_COORD_SCALE_XYZ

PAIRED_BALL_KEYS = (*MATCH_KEYS, "ball_observed", "player_observed")


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
        required = (*PAIRED_BALL_KEYS, "pred_ball_position")
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


def load_paired_ball_evaluations(
    baseline: Path, candidate: Path
) -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], dict[tuple[str, str], float], str, str
]:
    """Require identical teachers, masks, observations, windows, units and FPS."""
    base, fps, mode, split = _load(baseline)
    other, other_fps, other_mode, other_split = _load(candidate)
    if (fps, mode, split) != (other_fps, other_mode, other_split):
        raise ValueError("FPS, input condition or split mismatch")
    for key in PAIRED_BALL_KEYS:
        if base[key].dtype != other[key].dtype or not np.array_equal(
            base[key], other[key]
        ):
            raise ValueError(f"Unmatched paired array: {key}")
    return base, other, fps, mode, split
