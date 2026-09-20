"""Quantitative evaluation of an SLCS checkpoint on a dataset split.

Produces two artifacts:

- ``metrics.json``: aggregate metrics (BLCS/PLCS-comparable names, see
  :class:`src.tasks.slcs.training.metrics.SLCSMetrics`).
- ``eval_arrays.npz``: per-window, per-frame arrays (errors, masks,
  uncertainties, observation availability) consumed by the analysis script
  for error distributions, temporal error profiles, missing-rate breakdowns
  and confidence calibration.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from src.tasks.slcs.data.dataset import SLCSDataConfig, SLCSWindowDataset, collate_slcs
from src.tasks.slcs.evaluation.conditions import condition_inputs
from src.tasks.slcs.inference.predictor import SLCSPredictor
from src.tasks.slcs.training.metrics import SLCSMetrics
from src.utils.geometry.angles import angular_error
from src.utils.io import save_json
from src.utils.schema.court import COURT_COORD_SCALE_XYZ

SavezCompressed = Callable[..., None]


def evaluate_split(
    predictor: SLCSPredictor,
    *,
    dataset_root: str | Path,
    split_file: str | Path,
    split: str,
    data_config: SLCSDataConfig,
    batch_size: int,
    input_mode: str = "full",
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Evaluate a split; returns (aggregate metrics, per-frame arrays)."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    dataset = SLCSWindowDataset(
        dataset_root=dataset_root,
        split_file=split_file,
        split=split,
        config=data_config,
        augment=False,
        stride=(
            data_config.train_stride if split == "train" else data_config.eval_stride
        ),
    )
    metrics = SLCSMetrics()
    scale = torch.tensor(list(COURT_COORD_SCALE_XYZ), dtype=torch.float32)
    scale_mean = float(scale.mean().item())

    collected: dict[str, list[np.ndarray]] = {
        "pred_player_position": [],
        "pred_player_rotation": [],
        "pred_ball_position": [],
        "target_player_position": [],
        "target_player_rotation": [],
        "target_ball_position": [],
        "player_weight": [],
        "ball_weight": [],
        "frame_idx": [],
        "player_pos_error_m": [],
        "player_ang_error_deg": [],
        "ball_pos_error_m": [],
        "player_mask": [],
        "ball_mask": [],
        "padding_mask": [],
        "player_observed": [],
        "ball_observed": [],
        "player_sigma_m": [],
        "player_rot_sigma_deg": [],
        "ball_sigma_m": [],
    }

    for start in range(0, len(dataset), batch_size):
        samples = [
            dataset[i] for i in range(start, min(start + batch_size, len(dataset)))
        ]
        batch = condition_inputs(collate_slcs(samples), input_mode)
        outputs, targets = predictor.predict_with_targets(batch)

        padding_mask = targets.padding_mask
        player_mask = targets.player_mask
        ball_mask = targets.ball_mask
        metrics.update(outputs, targets)

        player_err = (
            (outputs.player_position - targets.target_player_position) * scale
        ).norm(dim=-1)
        ang_err = (
            angular_error(outputs.player_rotation, targets.target_player_rotation)
            * 180.0
            / math.pi
        )
        ball_err = (
            (outputs.ball_position - targets.target_ball_position) * scale
        ).norm(dim=-1)

        for key, value in {
            "pred_player_position": outputs.player_position,
            "pred_player_rotation": outputs.player_rotation,
            "pred_ball_position": outputs.ball_position,
            "target_player_position": targets.target_player_position,
            "target_player_rotation": targets.target_player_rotation,
            "target_ball_position": targets.target_ball_position,
            "player_weight": targets.player_weight,
            "ball_weight": targets.ball_weight,
        }.items():
            collected[key].append(value.numpy())
        frame_idx = batch["frame_idx"].numpy().copy()
        frame_idx[padding_mask.numpy()] = -1
        collected["frame_idx"].append(frame_idx)
        collected["player_pos_error_m"].append(player_err.numpy())
        collected["player_ang_error_deg"].append(ang_err.numpy())
        collected["ball_pos_error_m"].append(ball_err.numpy())
        collected["player_mask"].append(player_mask.numpy())
        collected["ball_mask"].append(ball_mask.numpy())
        collected["padding_mask"].append(padding_mask.numpy())
        collected["player_observed"].append((batch["player_valid"] > 0).numpy())
        collected["ball_observed"].append((batch["ball_vis"] > 0).numpy())
        collected["player_sigma_m"].append(
            (outputs.player_position_log_b.exp() * scale_mean).numpy()
        )
        collected["player_rot_sigma_deg"].append(
            (outputs.player_rotation_log_b.exp() * 180.0 / math.pi).numpy()
        )
        collected["ball_sigma_m"].append(
            (outputs.ball_position_log_b.exp() * scale_mean).numpy()
        )

    arrays = {key: np.concatenate(chunks, axis=0) for key, chunks in collected.items()}
    arrays["scene_ids"] = np.asarray(dataset.prediction_ids)
    for key, attribute in {
        "video_ids": "video_id",
        "clip_ids": "clip_id",
        "camera_ids": "camera_id",
        "window_start": "window_start",
        "window_length": "window_length",
    }.items():
        arrays[key] = np.asarray([getattr(meta, attribute) for meta in dataset.metas])
    report = metrics.compute()
    report["num_windows"] = float(len(dataset))
    return report, arrays


def evaluation_context(checkpoint: str | Path, *, input_mode: str) -> dict[str, Any]:
    """Describe artifact coordinates and reference-label interpretation explicitly."""
    digest = hashlib.sha256()
    with Path(checkpoint).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest.hexdigest(),
        "input_mode": input_mode,
        "reference_interpretation": (
            "Metrics measure agreement with dataset annotation targets. For pseudo-3D "
            "teacher annotations they measure teacher agreement, not independently "
            "measured 3D accuracy; reference quality is not verified by this evaluator."
        ),
        "position_representation": "normalized court coordinates (same as training payload)",
        "position_scale_xyz_m": list(COURT_COORD_SCALE_XYZ),
        "rotation_representation": "yaw (cos, sin)",
        "frame_idx_reference": "zero-based absolute frame in clip camera video; padding=-1",
        "padding_mask": "true indicates padding",
    }


def save_evaluation(
    output_dir: str | Path,
    report: dict[str, float],
    arrays: dict[str, np.ndarray],
    *,
    context: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write metrics.json and eval_arrays.npz into ``output_dir``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = dict(report)
    if context:
        payload["context"] = context
    metrics_path = save_json(payload, out / "metrics.json")
    arrays_path = out / "eval_arrays.npz"
    savez_compressed = cast(SavezCompressed, np.savez_compressed)
    savez_compressed(arrays_path, **arrays)
    return Path(metrics_path), arrays_path


__all__ = ["evaluate_split", "evaluation_context", "save_evaluation"]
