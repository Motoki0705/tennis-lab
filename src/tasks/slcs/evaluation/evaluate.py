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

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np

from src.tasks.base.data import CourtCoordinateContractMismatchError
from src.tasks.slcs.data.dataset import SLCSDataConfig, SLCSWindowDataset, collate_slcs
from src.tasks.slcs.inference.predictor import SLCSPredictor
from src.tasks.slcs.normalization import scalar_position_uncertainty_scale_m
from src.tasks.slcs.training.metrics import SLCSMetrics
from src.utils.geometry.angles import angular_error
from src.utils.io import save_json

SavezCompressed = Callable[..., None]


def evaluate_split(
    predictor: SLCSPredictor,
    *,
    dataset_root: str | Path,
    split_file: str | Path,
    split: str,
    data_config: SLCSDataConfig,
    batch_size: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Evaluate a split; returns (aggregate metrics, per-frame arrays)."""
    dataset = SLCSWindowDataset(
        dataset_root=dataset_root,
        split_file=split_file,
        split=split,
        config=data_config,
        stride=(
            data_config.train_stride if split == "train" else data_config.eval_stride
        ),
    )
    contract = data_config.court_coordinate_normalization
    if predictor.court_coordinate_normalization != contract:
        raise CourtCoordinateContractMismatchError(
            "SLCS evaluation dataset normalization "
            f"{contract.version!r}/{contract.scale_xyz!r} does not match "
            "predictor normalization "
            f"{predictor.court_coordinate_normalization.version!r}/"
            f"{predictor.court_coordinate_normalization.scale_xyz!r}."
        )
    metrics = SLCSMetrics(contract)
    uncertainty_scale_m = scalar_position_uncertainty_scale_m(contract)

    collected: dict[str, list[np.ndarray]] = {
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
        batch = collate_slcs(samples)
        outputs, targets = predictor.predict_with_targets(batch)

        padding_mask = targets.padding_mask
        player_mask = targets.player_mask
        ball_mask = targets.ball_mask
        metrics.update(outputs, targets)

        player_err = contract.denormalize_position(
            outputs.player_position - targets.target_player_position
        ).norm(dim=-1)
        ang_err = (
            angular_error(outputs.player_rotation, targets.target_player_rotation)
            * 180.0
            / math.pi
        )
        ball_err = contract.denormalize_position(
            outputs.ball_position - targets.target_ball_position
        ).norm(dim=-1)

        collected["player_pos_error_m"].append(player_err.numpy())
        collected["player_ang_error_deg"].append(ang_err.numpy())
        collected["ball_pos_error_m"].append(ball_err.numpy())
        collected["player_mask"].append(player_mask.numpy())
        collected["ball_mask"].append(ball_mask.numpy())
        collected["padding_mask"].append(padding_mask.numpy())
        collected["player_observed"].append((batch["player_valid"] > 0).numpy())
        collected["ball_observed"].append((batch["ball_vis"] > 0).numpy())
        collected["player_sigma_m"].append(
            (outputs.player_position_log_b.exp() * uncertainty_scale_m).numpy()
        )
        collected["player_rot_sigma_deg"].append(
            (outputs.player_rotation_log_b.exp() * 180.0 / math.pi).numpy()
        )
        collected["ball_sigma_m"].append(
            (outputs.ball_position_log_b.exp() * uncertainty_scale_m).numpy()
        )

    arrays = {key: np.concatenate(chunks, axis=0) for key, chunks in collected.items()}
    arrays["scene_ids"] = np.asarray(dataset.scenes)
    arrays["court_coordinate_normalization_version"] = np.asarray(contract.version)
    arrays["court_coordinate_scale_xyz_m"] = np.asarray(
        contract.scale_xyz, dtype=np.float32
    )
    report = metrics.compute()
    report["num_windows"] = float(len(dataset))
    return report, arrays


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


__all__ = ["evaluate_split", "save_evaluation"]
