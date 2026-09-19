"""Strict matched-window comparisons of SLCS pseudo-teacher agreement."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.tasks.slcs.model_io import SLCSDecodedOutput, SLCSTrainingTargets
from src.tasks.slcs.training.metrics import SLCSMetrics
from src.utils.schema.court import COURT_COORD_SCALE_XYZ

CONDITIONS = ("full", "no_rgb", "detector_gap", "rgb_only")
GAP_CONDITIONS = ("detector_gap", "detector_gap_no_rgb")
METRICS = ("player_position_error_m", "ball_position_error_m", "player_angular_error_deg")
MATCH_KEYS = (
    "scene_ids", "video_ids", "clip_ids", "camera_ids", "window_start", "window_length", "frame_idx",
    "target_player_position", "target_player_rotation", "target_ball_position",
    "player_mask", "ball_mask", "padding_mask", "player_weight", "ball_weight",
)
PREDICTION_KEYS = ("pred_player_position", "pred_player_rotation", "pred_ball_position")


def _validate_arrays(arrays: dict[str, np.ndarray]) -> None:
    for key in (*MATCH_KEYS, *PREDICTION_KEYS):
        if key not in arrays:
            raise ValueError(f"Missing evaluation array: {key}")
    mask = arrays["player_mask"]
    if mask.ndim != 3:
        raise ValueError("player_mask must have shape (windows, players, frames)")
    n, p, t = mask.shape
    shapes: dict[str, tuple[int, ...]] = {key: (n,) for key in MATCH_KEYS[:6]}
    shapes.update({"frame_idx": (n, t), "padding_mask": (n, t), "ball_mask": (n, t),
                   "ball_weight": (n, t), "player_mask": (n, p, t), "player_weight": (n, p, t)})
    for prefix in ("pred", "target"):
        shapes.update({f"{prefix}_player_position": (n, p, t, 3), f"{prefix}_player_rotation": (n, p, t, 2), f"{prefix}_ball_position": (n, t, 3)})
    for key, shape in shapes.items():
        value = arrays[key]
        if value.shape != shape:
            raise ValueError(f"Invalid shape for {key}: {value.shape}, expected {shape}")
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite evaluation array: {key}")
    if not n or len(set(arrays["scene_ids"].tolist())) != n:
        raise ValueError("scene_ids must be nonempty and unique")
    for key in ("player_mask", "ball_mask", "padding_mask"):
        if arrays[key].dtype != np.bool_:
            raise ValueError(f"{key} must be boolean")
    for key in ("scene_ids", "video_ids", "clip_ids", "camera_ids"):
        if arrays[key].dtype.kind != "U" or any(not value for value in arrays[key]):
            raise ValueError(f"{key} must contain nonempty string identifiers")
    for key in ("frame_idx", "window_start", "window_length"):
        if arrays[key].dtype.kind not in "iu":
            raise ValueError(f"{key} must contain integer indices")
    padding = arrays["padding_mask"]
    expected_padding = np.arange(t)[None] >= arrays["window_length"][:, None]
    expected_frames = arrays["window_start"][:, None] + np.arange(t)[None]
    expected_frames[expected_padding] = -1
    if (arrays["window_start"] < 0).any() or ((arrays["window_length"] < 1) | (arrays["window_length"] > t)).any():
        raise ValueError("Invalid window start or length")
    if not np.array_equal(padding, expected_padding) or not np.array_equal(arrays["frame_idx"], expected_frames):
        raise ValueError("Frame indices/padding disagree with window metadata")
    if (arrays["player_mask"] & padding[:, None]).any() or (arrays["ball_mask"] & padding).any():
        raise ValueError("Target masks must exclude padding")
    for entity in ("player", "ball"):
        weights = arrays[f"{entity}_weight"]
        if ((weights < 0) | (weights > 1)).any():
            raise ValueError("Weights must lie in [0, 1]")
        if (weights[~arrays[f"{entity}_mask"]] != 0).any():
            raise ValueError("Invalid target weights must be zero")


def _summarize(arrays: dict[str, np.ndarray], selection: np.ndarray) -> dict[str, float | int | None]:
    def tensor(key: str) -> torch.Tensor:
        return torch.from_numpy(arrays[key][selection])

    targets = SLCSTrainingTargets(
        target_player_position=tensor("target_player_position"), target_player_rotation=tensor("target_player_rotation"),
        target_ball_position=tensor("target_ball_position"), player_mask=tensor("player_mask"),
        player_weight=tensor("player_weight"), ball_mask=tensor("ball_mask"), ball_weight=tensor("ball_weight"), padding_mask=tensor("padding_mask"),
    )
    outputs = SLCSDecodedOutput(
        player_position=tensor("pred_player_position"), player_rotation=tensor("pred_player_rotation"), ball_position=tensor("pred_ball_position"),
        player_position_log_b=torch.zeros_like(targets.player_weight), player_rotation_log_b=torch.zeros_like(targets.player_weight),
        ball_position_log_b=torch.zeros_like(targets.ball_weight),
    )
    metrics = SLCSMetrics()
    metrics.update(outputs, targets)
    values = metrics.compute()
    result: dict[str, float | int | None] = {key: values.get(key) for key in METRICS}
    result["num_windows"] = int(selection.sum())
    for entity in ("player", "ball"):
        mask = arrays[f"{entity}_mask"][selection]
        weights = arrays[f"{entity}_weight"][selection][mask]
        result[f"{entity}_valid_count"] = int(mask.sum())
        result[f"{entity}_valid_weight_sum"] = float(weights.astype(np.float64).sum())
        result[f"{entity}_valid_weight_mean"] = float(weights.mean()) if weights.size else None
    return result


def compare_conditions(bundles: dict[str, Path], domains: dict[str, str]) -> dict[str, Any]:
    """Require exact paired samples/targets and identical checkpoint before comparing."""
    if set(bundles) != set(CONDITIONS):
        raise ValueError(f"Exactly these four conditions are required: {CONDITIONS}")
    return _compare_paired(bundles, domains, CONDITIONS, "full")


def compare_gap_conditions(bundles: dict[str, Path], domains: dict[str, str]) -> dict[str, Any]:
    """Measure RGB contribution with identical deterministic detector gaps."""
    if set(bundles) != set(GAP_CONDITIONS):
        raise ValueError(f"Exactly these gap conditions are required: {GAP_CONDITIONS}")
    return _compare_paired(bundles, domains, GAP_CONDITIONS, "detector_gap")


def _compare_paired(
    bundles: dict[str, Path], domains: dict[str, str], conditions: tuple[str, ...], reference: str
) -> dict[str, Any]:
    loaded: dict[str, dict[str, np.ndarray]] = {}
    checkpoint: str | None = None
    coordinate_context: dict[str, Any] | None = None
    for mode in conditions:
        payload = json.loads((bundles[mode] / "metrics.json").read_text())
        context = payload["context"]
        digest = context["checkpoint_sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid checkpoint SHA256")
        if context["input_mode"] != mode or (checkpoint is not None and checkpoint != digest):
            raise ValueError("Checkpoint SHA256 or input condition mismatch")
        checkpoint = digest
        coordinates = {key: context[key] for key in ("position_representation", "position_scale_xyz_m", "rotation_representation", "frame_idx_reference", "padding_mask")}
        if coordinates["position_scale_xyz_m"] != list(COURT_COORD_SCALE_XYZ) or coordinates["position_representation"] != "normalized court coordinates (same as training payload)" or coordinates["rotation_representation"] != "yaw (cos, sin)":
            raise ValueError("Unsupported coordinate representation")
        if coordinate_context is not None and coordinates != coordinate_context:
            raise ValueError("Coordinate context mismatch")
        coordinate_context = coordinates
        with np.load(bundles[mode] / "eval_arrays.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in (*MATCH_KEYS, *PREDICTION_KEYS)}
        _validate_arrays(arrays)
        if loaded:
            for key in MATCH_KEYS:
                base = loaded[reference][key]
                if base.dtype != arrays[key].dtype or not np.array_equal(base, arrays[key]):
                    raise ValueError(f"Unmatched {mode} array: {key}")
        loaded[mode] = arrays
    videos = loaded[reference]["video_ids"]
    if set(domains) != set(videos.tolist()) or any(not name for name in domains.values()):
        raise ValueError("Provide exactly one explicit domain for every evaluated video")
    domain_axis = np.array([domains[str(video)] for video in videos])
    groups: list[tuple[str, str, np.ndarray]] = [("all", "all", np.ones(len(videos), dtype=bool))]
    groups += [("video", str(video), videos == video) for video in np.unique(videos)]
    groups += [("domain", str(domain), domain_axis == domain) for domain in np.unique(domain_axis)]
    rows = []
    for group_type, group, selection in groups:
        scores = {mode: _summarize(loaded[mode], selection) for mode in conditions}
        for mode in conditions:
            row: dict[str, Any] = {"group_type": group_type, "group": group, "condition": mode, **scores[mode]}
            for metric in METRICS:
                full, other = scores[reference][metric], scores[mode][metric]
                row[f"{reference}_minus_condition_{metric}"] = None if full is None or other is None else full - other
            rows.append(row)
    return {
        "schema_version": 1, "checkpoint_sha256": checkpoint,
        "interpretation": f"Pseudo-teacher agreement; not measured 3D accuracy or a causal estimate. Negative {reference}-minus-condition error favors {reference}.",
        "aggregation": "SLCSMetrics masked unweighted means; label weights reported separately, not applied to headline metrics. Overlapping window occurrences remain separate.",
        "domain_mapping": domains, "sources": {key: str(value.resolve()) for key, value in bundles.items()},
        "coordinate_context": coordinate_context, "rows": rows,
    }


def save_comparison(report: dict[str, Any], output: Path) -> tuple[Path, Path]:
    """Write a new JSON/CSV report directory without replacing previous results."""
    output.mkdir(parents=True, exist_ok=False)
    json_path, csv_path = output / "comparison.json", output / "comparison.csv"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    rows = report["rows"]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path
