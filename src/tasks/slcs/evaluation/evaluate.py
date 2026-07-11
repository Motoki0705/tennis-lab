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
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.tasks.slcs.data.dataset import SLCSDataConfig, SLCSWindowDataset, collate_slcs
from src.tasks.slcs.inference.predictor import SLCSPredictor
from src.tasks.slcs.training.metrics import SLCSMetrics
from src.utils.geometry.angles import angular_error
from src.utils.io import save_json
from src.utils.schema.court import COURT_COORD_SCALE_XYZ


def evaluate_split(
    predictor: SLCSPredictor,
    *,
    dataset_root: str | Path,
    split_file: str | Path,
    split: str,
    data_config: SLCSDataConfig,
    batch_size: int = 4,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Evaluate a split; returns (aggregate metrics, per-frame arrays)."""
    dataset = SLCSWindowDataset(
        dataset_root=dataset_root,
        split_file=split_file,
        split=split,
        config=data_config,
    )
    metrics = SLCSMetrics()
    scale = torch.tensor(list(COURT_COORD_SCALE_XYZ), dtype=torch.float32)
    scale_mean = float(scale.mean().item())

    collected: dict[str, list[np.ndarray]] = {
        "player_pos_error_m": [],
        "player_ang_error_deg": [],
        "ball_pos_error_m": [],
        "player_mask": [],
        "ball_mask": [],
        "frame_mask": [],
        "player_observed": [],
        "ball_observed": [],
        "player_sigma_m": [],
        "player_rot_sigma_deg": [],
        "ball_sigma_m": [],
    }

    for start in range(0, len(dataset), batch_size):
        samples = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        batch = collate_slcs(samples)
        outputs = predictor.predict(batch, denormalize=False)

        frame_mask = batch["frame_mask"] > 0
        player_mask = (batch["target_player_valid"] > 0) & frame_mask.unsqueeze(1)
        ball_mask = (batch["target_ball_valid"] > 0) & frame_mask
        metrics.update(outputs, batch)

        player_err = (
            (outputs["player_position"] - batch["target_player_position"]) * scale
        ).norm(dim=-1)
        ang_err = (
            angular_error(outputs["player_rotation"], batch["target_player_rotation"])
            * 180.0
            / math.pi
        )
        ball_err = ((outputs["ball_position"] - batch["target_ball_position"]) * scale).norm(
            dim=-1
        )

        collected["player_pos_error_m"].append(player_err.numpy())
        collected["player_ang_error_deg"].append(ang_err.numpy())
        collected["ball_pos_error_m"].append(ball_err.numpy())
        collected["player_mask"].append(player_mask.numpy())
        collected["ball_mask"].append(ball_mask.numpy())
        collected["frame_mask"].append(frame_mask.numpy())
        collected["player_observed"].append((batch["player_valid"] > 0).numpy())
        collected["ball_observed"].append((batch["ball_vis"] > 0).numpy())
        collected["player_sigma_m"].append(
            (outputs["player_position_log_b"].exp() * scale_mean).numpy()
        )
        collected["player_rot_sigma_deg"].append(
            (outputs["player_rotation_log_b"].exp() * 180.0 / math.pi).numpy()
        )
        collected["ball_sigma_m"].append(
            (outputs["ball_position_log_b"].exp() * scale_mean).numpy()
        )

    arrays = {key: np.concatenate(chunks, axis=0) for key, chunks in collected.items()}
    arrays["scene_ids"] = np.asarray(dataset.scenes)
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
    np.savez_compressed(arrays_path, **arrays)  # type: ignore[arg-type]
    return Path(metrics_path), arrays_path


__all__ = ["evaluate_split", "save_evaluation"]
