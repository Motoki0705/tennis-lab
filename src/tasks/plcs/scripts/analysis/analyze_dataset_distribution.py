"""Analyze PLCS dataset distributions (position, yaw, cameras) using Hydra.

This script inspects pre-generated PLCS scene NPZ files and summarizes:
- Player position distribution (per-frame and initial conditions)
- Player yaw distribution
- Camera count distribution (after filtering)
- Fractions near the court origin (useful for detecting dataset bias)

Example commands:
    `uv run python -m src.tasks.plcs.scripts.analysis.analyze_dataset_distribution`
    `uv run python -m src.tasks.plcs.scripts.analysis.analyze_dataset_distribution run.output_dir=outputs/plcs/analysis/dataset_distribution analysis.max_scenes=200`

Config entry point: `src/tasks/plcs/configs/analyze_dataset_distribution.yaml`
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src.utils.schema.court import COURT_COORD_SCALE_XYZ

F = TypeVar("F", bound=Callable[..., Any])


def hydra_main(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    """Typed wrapper for hydra.main to keep mypy satisfied."""
    return cast(Callable[[F], F], hydra.main(*args, **kwargs))

@dataclass
class RunningStats:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size == 0:
            return

        batch_n = int(values.size)
        batch_mean = float(values.mean())
        batch_var = float(values.var(ddof=0))
        batch_m2 = batch_var * batch_n

        self.min = min(self.min, float(values.min()))
        self.max = max(self.max, float(values.max()))

        if self.n == 0:
            self.n = batch_n
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        delta = batch_mean - self.mean
        n_total = self.n + batch_n
        self.mean = self.mean + delta * batch_n / n_total
        self.m2 = self.m2 + batch_m2 + (delta * delta) * self.n * batch_n / n_total
        self.n = n_total

    def to_dict(self) -> dict[str, float | int | None]:
        if self.n == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
        var = self.m2 / self.n
        return {
            "count": self.n,
            "mean": float(self.mean),
            "std": float(math.sqrt(max(var, 0.0))),
            "min": float(self.min),
            "max": float(self.max),
        }


def _prepare_paths(cfg: DictConfig) -> DictConfig:
    cfg.run.output_dir = to_absolute_path(str(cfg.run.output_dir))
    cfg.data.scene_dir = to_absolute_path(str(cfg.data.scene_dir))
    return cfg


def _decode_npz_json(item: Any) -> Any:
    if isinstance(item, (bytes, bytearray)):
        item = item.decode("utf-8")
    if isinstance(item, str):
        return json.loads(item)
    return item


def _iter_scene_files(scene_dir: Path) -> list[Path]:
    scenes_subdir = scene_dir / "scenes"
    files = sorted(scenes_subdir.glob("scene_*.npz"))
    if not files:
        raise ValueError(f"No scene files found in {scenes_subdir}")
    return files


def _safe_initial_xyz(meta: dict[str, Any]) -> tuple[float, float, float]:
    init = meta.get("initial_position")
    if init is None:
        return (float("nan"), float("nan"), float("nan"))
    if isinstance(init, (list, tuple)) and len(init) >= 2:
        x = float(init[0])
        y = float(init[1])
        z = float(init[2]) if len(init) >= 3 else 0.0
        return (x, y, z)
    return (float("nan"), float("nan"), float("nan"))


def _circular_mean_from_sincos(sum_sin: float, sum_cos: float) -> float | None:
    if sum_sin == 0.0 and sum_cos == 0.0:
        return None
    return float(math.atan2(sum_sin, sum_cos))


@hydra_main(
    config_path="../../configs",
    config_name="analyze_dataset_distribution",
    version_base="1.3",
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry
    cfg = _prepare_paths(cfg)

    seed = int(cfg.run.seed)
    random.seed(seed)
    np.random.seed(seed)

    scene_dir = Path(str(cfg.data.scene_dir))
    out_dir = Path(str(cfg.run.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")

    scene_files = _iter_scene_files(scene_dir)
    max_scenes = cfg.analysis.max_scenes
    if max_scenes is not None:
        max_scenes = int(max_scenes)
        if max_scenes < len(scene_files):
            scene_files = random.sample(scene_files, k=max_scenes)
            scene_files = sorted(scene_files)

    mode = str(cfg.analysis.mode)
    if mode not in {"per_frame", "initial_only"}:
        raise ValueError(f"Unknown analysis.mode: {mode}")
    max_frames_per_scene = cfg.analysis.max_frames_per_scene
    if max_frames_per_scene is not None:
        max_frames_per_scene = int(max_frames_per_scene)

    scale_xyz = np.asarray(COURT_COORD_SCALE_XYZ, dtype=np.float64)  # (3,)

    x_stats = RunningStats()
    y_stats = RunningStats()
    z_stats = RunningStats()
    xy_radius_stats = RunningStats()
    xyz_radius_stats = RunningStats()

    yaw_stats = RunningStats()
    yaw_sum_sin = 0.0
    yaw_sum_cos = 0.0
    yaw_count = 0

    num_cameras_stats = RunningStats()

    thresholds = [float(t) for t in cfg.analysis.radius_thresholds_m]
    xy_within_counts = {t: 0 for t in thresholds}
    xyz_within_counts = {t: 0 for t in thresholds}

    xy_hist_cfg = cfg.analysis.xy_hist
    x_range = tuple(float(v) for v in xy_hist_cfg.x_range_m)
    y_range = tuple(float(v) for v in xy_hist_cfg.y_range_m)
    bins_x = int(xy_hist_cfg.bins_x)
    bins_y = int(xy_hist_cfg.bins_y)

    yaw_hist_cfg = cfg.analysis.yaw_hist
    yaw_bins = int(yaw_hist_cfg.bins)
    yaw_range = (-math.pi, math.pi)

    xy_hist: np.ndarray = np.zeros((bins_x, bins_y), dtype=np.int64)
    yaw_hist: np.ndarray = np.zeros((yaw_bins,), dtype=np.int64)
    x_edges = np.linspace(x_range[0], x_range[1], bins_x + 1, dtype=np.float64)
    y_edges = np.linspace(y_range[0], y_range[1], bins_y + 1, dtype=np.float64)
    yaw_edges = np.linspace(yaw_range[0], yaw_range[1], yaw_bins + 1, dtype=np.float64)

    initial_csv_path = out_dir / "initial_conditions.csv"
    with open(initial_csv_path, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(
            [
                "scene_id",
                "init_x_m",
                "init_y_m",
                "init_z_m",
                "init_yaw_rad",
                "num_cameras",
                "num_frames",
            ]
        )

        total_frames_used = 0
        total_scenes_used = 0

        for scene_path in scene_files:
            with np.load(scene_path, allow_pickle=True) as data:
                meta = _decode_npz_json(data["meta"].item())
                if not isinstance(meta, dict):
                    raise ValueError(f"Invalid meta in {scene_path}")

                scene_id = str(meta.get("scene_id", scene_path.stem))
                init_x, init_y, init_z = _safe_initial_xyz(meta)
                init_yaw = float(meta.get("initial_yaw", float("nan")))

                num_frames = int(meta.get("num_frames", int(data["position"].shape[0])))
                if "num_cameras" in data.files:
                    num_cameras = int(np.asarray(data["num_cameras"]).item())
                else:
                    num_cameras = 0

                writer.writerow(
                    [
                        scene_id,
                        init_x,
                        init_y,
                        init_z,
                        init_yaw,
                        num_cameras,
                        num_frames,
                    ]
                )

                num_cameras_stats.update(np.asarray([num_cameras], dtype=np.float64))

                if mode == "initial_only":
                    total_scenes_used += 1
                    continue

                pos_norm = np.asarray(data["position"], dtype=np.float64)  # (T, 3)
                rot = np.asarray(data["rotation"], dtype=np.float64)  # (T, 2)

                if max_frames_per_scene is not None and pos_norm.shape[0] > max_frames_per_scene:
                    idx = np.random.choice(pos_norm.shape[0], size=max_frames_per_scene, replace=False)
                    pos_norm = pos_norm[idx]
                    rot = rot[idx]

                pos_m = pos_norm * scale_xyz
                x = pos_m[:, 0]
                y = pos_m[:, 1]
                z = pos_m[:, 2]

                xy_r = np.sqrt(x * x + y * y)
                xyz_r = np.sqrt(x * x + y * y + z * z)

                x_stats.update(x)
                y_stats.update(y)
                z_stats.update(z)
                xy_radius_stats.update(xy_r)
                xyz_radius_stats.update(xyz_r)

                for t in thresholds:
                    xy_within_counts[t] += int((xy_r <= t).sum())
                    xyz_within_counts[t] += int((xyz_r <= t).sum())

                yaw = np.arctan2(rot[:, 0], rot[:, 1])
                yaw_stats.update(yaw)
                yaw_sum_sin += float(np.sin(yaw).sum())
                yaw_sum_cos += float(np.cos(yaw).sum())
                yaw_count += int(yaw.size)

                h2d, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
                xy_hist += h2d.astype(np.int64)

                h1d, _ = np.histogram(yaw, bins=yaw_edges)
                yaw_hist += h1d.astype(np.int64)

                total_frames_used += int(pos_m.shape[0])
                total_scenes_used += 1

    xy_within_frac = (
        {str(t): (xy_within_counts[t] / total_frames_used if total_frames_used else None) for t in thresholds}
        if mode != "initial_only"
        else {}
    )
    xyz_within_frac = (
        {str(t): (xyz_within_counts[t] / total_frames_used if total_frames_used else None) for t in thresholds}
        if mode != "initial_only"
        else {}
    )

    summary = {
        "scene_dir": str(scene_dir),
        "output_dir": str(out_dir),
        "mode": mode,
        "scenes_used": total_scenes_used,
        "frames_used": total_frames_used,
        "num_cameras": num_cameras_stats.to_dict(),
        "position_m": {
            "x": x_stats.to_dict(),
            "y": y_stats.to_dict(),
            "z": z_stats.to_dict(),
            "xy_radius": xy_radius_stats.to_dict(),
            "xyz_radius": xyz_radius_stats.to_dict(),
        },
        "yaw_rad": {
            "linear": yaw_stats.to_dict(),
            "circular_mean": _circular_mean_from_sincos(yaw_sum_sin, yaw_sum_cos),
            "count": yaw_count,
        },
        "fractions": {
            "xy_radius_le_m": xy_within_frac,
            "xyz_radius_le_m": xyz_within_frac,
        },
        "hist_files": {
            "xy_hist_npz": "hist_xy.npz",
            "yaw_hist_npz": "hist_yaw.npz",
        },
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    np.savez_compressed(out_dir / "hist_xy.npz", hist=xy_hist, x_edges=x_edges, y_edges=y_edges)
    np.savez_compressed(out_dir / "hist_yaw.npz", hist=yaw_hist, yaw_edges=yaw_edges)

    if bool(cfg.plots.enabled):
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 5))
            im = ax.imshow(
                xy_hist.T,
                origin="lower",
                extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                aspect="auto",
            )
            ax.set_title("PLCS position histogram (meters): x vs y")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            fig.colorbar(im, ax=ax, label="count")
            fig.tight_layout()
            fig.savefig(out_dir / "xy_hist.png", dpi=int(cfg.plots.dpi))
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(7, 3))
            centers = 0.5 * (yaw_edges[:-1] + yaw_edges[1:])
            ax.plot(centers, yaw_hist, linewidth=1.5)
            ax.set_title("PLCS yaw histogram (radians)")
            ax.set_xlabel("yaw (rad)")
            ax.set_ylabel("count")
            fig.tight_layout()
            fig.savefig(out_dir / "yaw_hist.png", dpi=int(cfg.plots.dpi))
            plt.close(fig)
        except Exception as exc:  # pragma: no cover - optional plotting
            print(f"Plotting skipped: {exc}")

    print(f"Wrote: {out_dir / 'summary.json'}")
    print(f"Wrote: {initial_csv_path}")
    print(f"Wrote: {out_dir / 'hist_xy.npz'}")
    print(f"Wrote: {out_dir / 'hist_yaw.npz'}")
    if mode != "initial_only":
        print("Fractions near origin (XY radius):")
        for t in thresholds:
            frac = xy_within_frac.get(str(t))
            print(f"  r_xy <= {t:.2f} m: {frac}")

    return 0


if __name__ == "__main__":
    raise SystemExit(cast(Callable[[], int], main)())
