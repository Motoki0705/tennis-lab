"""Masked physical-coordinate variance and input sensitivity for selected checkpoints."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.utils.schema.court import COURT_COORD_SCALE_XYZ


def main() -> None:
    root = Path("/home/kamimura/projects/tennis-lab/outputs")
    result: dict[str, Any] = {"interpretation": "Pseudo-teacher agreement; low predicted variance is a diagnostic, not proof of a constant predictor. All ball statistics use identical teacher-valid masks. RMS3D=sqrt(mean(sum(delta_xyz**2)))."}
    scale = np.asarray(COURT_COORD_SCALE_XYZ, np.float32)
    for model in ("baseline", "augmented"):
        model_report: dict[str, Any] = {}
        for split in ("val", "test"):
            arrays = {}
            for condition in ("full", "no_rgb", "detector_gap"):
                directory = root / f"slcs/evaluate/real_rgb_pilot_{model}_{split}_{condition}/s42-002"
                with np.load(directory / "eval_arrays.npz") as data:
                    arrays[condition] = {key: data[key] for key in data.files}
            full = arrays["full"]
            groups: dict[str, np.ndarray] = {"all": np.ones(len(full["clip_ids"]), dtype=bool)}
            groups.update({str(clip): full["clip_ids"] == clip for clip in np.unique(full["clip_ids"])})
            summaries = {}
            for group, selection in groups.items():
                mask = full["ball_mask"][selection]
                pred = (full["pred_ball_position"][selection] * scale)[mask]
                target = (full["target_ball_position"][selection] * scale)[mask]
                summary: dict[str, Any] = {"valid_count": len(pred), "pred_mean_xyz_m": pred.mean(0).tolist(), "pred_std_xyz_m": pred.std(0).tolist(), "target_mean_xyz_m": target.mean(0).tolist(), "target_std_xyz_m": target.std(0).tolist(), "ball_error_m": float(np.linalg.norm(pred - target, axis=-1).mean())}
                summary["pred_std_norm_over_target_std_norm"] = float(np.linalg.norm(pred.std(0)) / np.linalg.norm(target.std(0)))
                for mode in ("no_rgb", "detector_gap"):
                    delta = pred - (arrays[mode]["pred_ball_position"][selection] * scale)[mask]
                    summary[f"full_minus_{mode}_rms3d_m"] = float(np.sqrt(np.square(delta).sum(-1).mean()))
                    summary[f"full_minus_{mode}_rms_xyz_m"] = np.sqrt(np.square(delta).mean(0)).tolist()
                summaries[group] = summary
            model_report[split] = summaries
        events = EventAccumulator(str(root / f"slcs/train/real_rgb_pilot_{model}/s42-002/logs/version_0"), size_guidance={"scalars": 0})
        events.Reload()
        model_report["training_history"] = {}
        for tag in ("train/player_position_error_m_epoch", "train/ball_position_error_m_epoch", "val/player_position_error_m_epoch", "val/ball_position_error_m_epoch"):
            values = events.Scalars(tag)
            model_report["training_history"][tag] = {"first": {"step": values[0].step, "value": values[0].value}, "last": {"step": values[-1].step, "value": values[-1].value}, "last_five": [{"step": item.step, "value": item.value} for item in values[-5:]]}
        result[model] = model_report
    destination = root / "slcs/analyze/real_rgb_pilot_comparison/s42-002/ball_diagnostics.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(destination)


if __name__ == "__main__":
    main()
