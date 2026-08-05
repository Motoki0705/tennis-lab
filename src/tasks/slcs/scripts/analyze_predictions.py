"""
Analyze SLCS evaluation arrays: error distributions, temporal error profile,
observation-missing-rate breakdown and confidence calibration.

Usage:
    python -m src.tasks.slcs.scripts.analyze_predictions analysis.arrays=slcs/evaluate/run/eval_arrays.npz
    python -m src.tasks.slcs.scripts.analyze_predictions analysis.arrays=... analysis.output_dir=slcs/analysis/run1
    python -m src.tasks.slcs.scripts.analyze_predictions analysis.arrays=... analysis.calibration_bins=8

Notes:
    - Configuration is loaded from `src/tasks/slcs/configs/analyze_predictions.yaml`.
    - Input and output paths are relative to `paths.output_root`.
    - Input is the `eval_arrays.npz` produced by `scripts/evaluate.py`.
    - Writes PNG plots (error histograms, error-vs-window-offset, observed vs
      missing observation breakdown, sigma reliability curves) plus a
      machine-readable `analysis.json`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from src.tasks.slcs.configuration import SLCSAnalysisConfig
from src.utils.hydra import hydra_main
from src.utils.io import save_json


def _masked_values(values: NDArray[Any], mask: NDArray[Any]) -> NDArray[np.float64]:
    selected = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    return np.asarray(selected, dtype=np.float64)


def _temporal_profile(
    errors: NDArray[Any], mask: NDArray[Any], time_axis: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mean error and coverage per window time offset."""
    err = np.asarray(errors, dtype=np.float64)
    msk = np.asarray(mask, dtype=np.float64)
    axes = tuple(i for i in range(err.ndim) if i != time_axis)
    weight = msk.sum(axis=axes)
    total = (err * msk).sum(axis=axes)
    profile = np.divide(total, weight, out=np.zeros_like(total), where=weight > 0)
    return profile, weight


def _calibration_curve(
    sigma: NDArray[np.float64], error: NDArray[np.float64], bins: int
) -> dict[str, list[float]]:
    """Bin frames by predicted sigma; report mean sigma vs mean error per bin."""
    if sigma.size == 0:
        return {"bin_sigma": [], "bin_error": [], "bin_count": []}
    order = np.argsort(sigma)
    sigma_sorted, error_sorted = sigma[order], error[order]
    splits = np.array_split(np.arange(sigma.size), bins)
    out: dict[str, list[float]] = {"bin_sigma": [], "bin_error": [], "bin_count": []}
    for idx in splits:
        if idx.size == 0:
            continue
        out["bin_sigma"].append(float(sigma_sorted[idx].mean()))
        out["bin_error"].append(float(error_sorted[idx].mean()))
        out["bin_count"].append(float(idx.size))
    return out


def _plot_histograms(data: dict[str, NDArray[np.float64]], path: Path) -> None:
    fig, axes = plt.subplots(1, len(data), figsize=(5 * len(data), 4))
    if len(data) == 1:
        axes = [axes]
    for ax, (title, values) in zip(axes, data.items(), strict=True):
        if values.size:
            ax.hist(values, bins=40, color="tab:blue", alpha=0.8)
            ax.axvline(
                float(np.median(values)),
                color="tab:red",
                linestyle="--",
                label=f"median={np.median(values):.3f}",
            )
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("frames")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(config: DictConfig) -> None:
    """Run the analysis and write plots + analysis.json."""
    runtime = SLCSAnalysisConfig.from_config(config)
    arrays = np.load(runtime.arrays, allow_pickle=False)
    output_dir = runtime.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    bins = runtime.calibration_bins

    player_mask = arrays["player_mask"]
    ball_mask = arrays["ball_mask"]
    frame_mask = arrays["frame_mask"]

    player_err = _masked_values(arrays["player_pos_error_m"], player_mask)
    player_ang = _masked_values(arrays["player_ang_error_deg"], player_mask)
    ball_err = _masked_values(arrays["ball_pos_error_m"], ball_mask)

    report: dict[str, Any] = {
        "num_windows": int(frame_mask.shape[0]),
        "player_frames": int(player_mask.sum()),
        "ball_frames": int(ball_mask.sum()),
        "label_missing_rate_player": float(
            1.0 - player_mask.sum() / max(frame_mask.sum() * player_mask.shape[1], 1)
        ),
        "label_missing_rate_ball": float(
            1.0 - ball_mask.sum() / max(frame_mask.sum(), 1)
        ),
    }

    # 1. Error distributions ------------------------------------------------
    _plot_histograms(
        {
            "player position error [m]": player_err,
            "player yaw error [deg]": player_ang,
            "ball position error [m]": ball_err,
        },
        output_dir / "error_histograms.png",
    )
    for name, values in (
        ("player_pos_error_m", player_err),
        ("player_ang_error_deg", player_ang),
        ("ball_pos_error_m", ball_err),
    ):
        if values.size:
            report[name] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "p90": float(np.percentile(values, 90)),
            }

    # 2. Temporal error profile (window offset) -----------------------------
    player_profile, player_cov = _temporal_profile(
        arrays["player_pos_error_m"], player_mask, time_axis=2
    )
    ball_profile, ball_cov = _temporal_profile(
        arrays["ball_pos_error_m"], ball_mask, time_axis=1
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(player_profile, label="player position error [m]")
    ax.plot(ball_profile, label="ball position error [m]")
    ax.set_xlabel("window frame offset")
    ax.set_ylabel("mean error [m]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "temporal_error_profile.png")
    plt.close(fig)
    report["temporal_profile"] = {
        "player_pos_error_m": player_profile.tolist(),
        "player_coverage": player_cov.tolist(),
        "ball_pos_error_m": ball_profile.tolist(),
        "ball_coverage": ball_cov.tolist(),
    }

    # 3. Observation availability vs error ----------------------------------
    player_observed = arrays["player_observed"].astype(bool)
    ball_observed = arrays["ball_observed"].astype(bool)
    breakdown: dict[str, dict[str, float]] = {}
    for name, err_arr, mask_arr, obs_arr in (
        ("player_position", arrays["player_pos_error_m"], player_mask, player_observed),
        (
            "ball_position",
            arrays["ball_pos_error_m"],
            ball_mask,
            ball_observed,
        ),
    ):
        mask = np.asarray(mask_arr, dtype=bool)
        observed = _masked_values(err_arr, mask & obs_arr)
        missing = _masked_values(err_arr, mask & ~obs_arr)
        breakdown[name] = {
            "observed_mean": float(observed.mean()) if observed.size else float("nan"),
            "observed_frames": int(observed.size),
            "missing_mean": float(missing.mean()) if missing.size else float("nan"),
            "missing_frames": int(missing.size),
        }
    report["observation_breakdown"] = breakdown
    fig, ax = plt.subplots(figsize=(6, 4))
    labels, observed_means, missing_means = [], [], []
    for name, stats in breakdown.items():
        labels.append(name)
        observed_means.append(stats["observed_mean"])
        missing_means.append(stats["missing_mean"])
    x = np.arange(len(labels))
    ax.bar(x - 0.2, observed_means, width=0.4, label="2D observed")
    ax.bar(x + 0.2, missing_means, width=0.4, label="2D missing")
    ax.set_xticks(x, labels)
    ax.set_ylabel("mean error [m]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "observation_breakdown.png")
    plt.close(fig)

    # 4. Confidence calibration ---------------------------------------------
    calibration = {
        "player_position": _calibration_curve(
            _masked_values(arrays["player_sigma_m"], player_mask), player_err, bins
        ),
        "player_rotation": _calibration_curve(
            _masked_values(arrays["player_rot_sigma_deg"], player_mask),
            player_ang,
            bins,
        ),
        "ball_position": _calibration_curve(
            _masked_values(arrays["ball_sigma_m"], ball_mask), ball_err, bins
        ),
    }
    report["calibration"] = calibration
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, curve) in zip(axes, calibration.items(), strict=True):
        if curve["bin_sigma"]:
            ax.plot(curve["bin_sigma"], curve["bin_error"], marker="o")
            lim = max(max(curve["bin_sigma"]), max(curve["bin_error"]))
            ax.plot([0, lim], [0, lim], linestyle="--", color="gray", alpha=0.6)
        ax.set_title(f"{name} reliability", fontsize=9)
        ax.set_xlabel("predicted sigma")
        ax.set_ylabel("actual mean error")
    fig.tight_layout()
    fig.savefig(output_dir / "confidence_calibration.png")
    plt.close(fig)

    report_path = save_json(report, output_dir / "analysis.json")
    print(f"analysis -> {report_path}")


@hydra_main(
    config_path="../configs",
    config_name="analyze_predictions",
    version_base="1.3",
    validation_boundary="slcs.analyze_predictions",
)
def main(config: DictConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point for prediction analysis."""
    run(config)


if __name__ == "__main__":
    main()
