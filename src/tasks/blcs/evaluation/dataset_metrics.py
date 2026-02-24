"""Evaluate BLCS dataset trajectory distributions.

Computes per-shot metrics from NPZ scene files, including:
- apex height (m) and time to apex (s)
- time to first bounce (s)
- net clearance (m) and net-cross height (m) at y=0 crossing
- horizontal range (m) and flight path length (m)
- average speed (m/s) and speed at bounce (m/s)

Supports both single-shot scenes and rally scenes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from src.utils.schema.court import HALF_DOUBLES_WIDTH, NET_HEIGHT_CENTER, NET_HEIGHT_POST


@dataclass(frozen=True)
class MetricSummary:
    """Summary statistics for a list of floats."""

    count: int
    mean: float
    std: float
    min: float
    max: float
    p50: float
    p90: float
    p95: float

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "p50": self.p50,
            "p90": self.p90,
            "p95": self.p95,
        }


def _net_height_at_x(x: float) -> float:
    x_ratio = min(1.0, abs(x) / float(HALF_DOUBLES_WIDTH))
    return float(NET_HEIGHT_CENTER + x_ratio * (NET_HEIGHT_POST - NET_HEIGHT_CENTER))


def _net_clearance_m(positions: np.ndarray) -> float | None:
    """Approximate net clearance from a trajectory segment (output fps)."""
    if positions.size == 0:
        return None
    y = positions[:, 1]
    sign = np.sign(y)
    for i in range(1, len(y)):
        if sign[i - 1] == 0.0:
            x_at = float(positions[i - 1, 0])
            z_at = float(positions[i - 1, 2])
            return z_at - _net_height_at_x(x_at)
        if sign[i] == 0.0 or sign[i - 1] != sign[i]:
            p0 = positions[i - 1]
            p1 = positions[i]
            y0 = float(p0[1])
            y1 = float(p1[1])
            t = y0 / (y0 - y1 + 1e-8)
            x_at = float(p0[0] + t * (p1[0] - p0[0]))
            z_at = float(p0[2] + t * (p1[2] - p0[2]))
            return z_at - _net_height_at_x(x_at)
    return None


def _net_cross_height_m(positions: np.ndarray) -> float | None:
    """Return z at y=0 crossing if the trajectory crosses the net."""
    if positions.size == 0:
        return None
    y = positions[:, 1]
    sign = np.sign(y)
    for i in range(1, len(y)):
        if sign[i - 1] == 0.0:
            return float(positions[i - 1, 2])
        if sign[i] == 0.0 or sign[i - 1] != sign[i]:
            p0 = positions[i - 1]
            p1 = positions[i]
            y0 = float(p0[1])
            y1 = float(p1[1])
            t = y0 / (y0 - y1 + 1e-8)
            return float(p0[2] + t * (p1[2] - p0[2]))
    return None


def _summary(values: list[float]) -> MetricSummary | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return MetricSummary(
        count=int(arr.size),
        mean=float(arr.mean()),
        std=float(arr.std(ddof=0)),
        min=float(arr.min()),
        max=float(arr.max()),
        p50=float(np.percentile(arr, 50)),
        p90=float(np.percentile(arr, 90)),
        p95=float(np.percentile(arr, 95)),
    )


def _iter_npz_files(root: Path, max_files: int | None) -> Iterable[Path]:
    files = sorted(root.rglob("*.npz"))
    if max_files is not None:
        files = files[: max_files]
    return files


def _collect_shot_metrics(
    meta: dict, positions: np.ndarray, velocities: np.ndarray
) -> list[dict]:
    fps_out = int(meta["fps_out"])
    metrics = []

    if "shots" not in meta:
        # Single-shot scene
        t_start = 0
        t_bounce1 = int(meta["t_bounce1"])
        category = meta.get("category", "unknown")
        metrics.append(
            _compute_metrics_for_shot(
                positions=positions,
                velocities=velocities,
                t_start=t_start,
                t_bounce1=t_bounce1,
                fps_out=fps_out,
                category=category,
            )
        )
        return metrics

    for shot in meta["shots"]:
        t_start = int(shot["t_start"])
        t_bounce1 = int(shot["t_bounce1"])
        category = shot.get("category", "unknown")
        metrics.append(
            _compute_metrics_for_shot(
                positions=positions,
                velocities=velocities,
                t_start=t_start,
                t_bounce1=t_bounce1,
                fps_out=fps_out,
                category=category,
            )
        )
    return metrics


def _compute_metrics_for_shot(
    positions: np.ndarray,
    velocities: np.ndarray,
    t_start: int,
    t_bounce1: int,
    fps_out: int,
    category: str,
) -> dict:
    if t_bounce1 < 0 or t_start < 0 or t_start >= len(positions):
        return {
            "category": category,
            "apex_height_m": None,
            "apex_time_s": None,
            "time_to_bounce1_s": None,
            "net_clearance_m": None,
            "net_cross_height_m": None,
            "horizontal_range_m": None,
            "flight_path_m": None,
            "avg_speed_mps": None,
            "speed_at_bounce_mps": None,
        }

    end = min(t_bounce1 + 1, len(positions))
    segment = positions[t_start:end]
    if segment.size == 0:
        return {
            "category": category,
            "apex_height_m": None,
            "apex_time_s": None,
            "time_to_bounce1_s": None,
            "net_clearance_m": None,
            "net_cross_height_m": None,
            "horizontal_range_m": None,
            "flight_path_m": None,
            "avg_speed_mps": None,
            "speed_at_bounce_mps": None,
        }

    apex = float(segment[:, 2].max())
    apex_idx = int(segment[:, 2].argmax())
    apex_time = float(apex_idx) / float(fps_out)
    time_to_bounce = float(t_bounce1 - t_start) / float(fps_out)
    net_clearance = _net_clearance_m(segment)
    net_cross_height = _net_cross_height_m(segment)

    start_xy = segment[0, :2]
    end_xy = segment[-1, :2]
    horizontal_range = float(np.linalg.norm(end_xy - start_xy))

    diffs = np.diff(segment, axis=0)
    flight_path = float(np.linalg.norm(diffs, axis=1).sum()) if len(diffs) > 0 else 0.0

    vel_segment = velocities[t_start:end]
    speed_segment = np.linalg.norm(vel_segment, axis=1) if vel_segment.size > 0 else None
    avg_speed = float(speed_segment.mean()) if speed_segment is not None else None
    speed_at_bounce = (
        float(speed_segment[-1])
        if speed_segment is not None and len(speed_segment) > 0
        else None
    )

    return {
        "category": category,
        "apex_height_m": apex,
        "apex_time_s": apex_time,
        "time_to_bounce1_s": time_to_bounce,
        "net_clearance_m": net_clearance,
        "net_cross_height_m": net_cross_height,
        "horizontal_range_m": horizontal_range,
        "flight_path_m": flight_path,
        "avg_speed_mps": avg_speed,
        "speed_at_bounce_mps": speed_at_bounce,
    }


def evaluate_dataset(
    root: Path,
    max_files: int | None = None,
    lob_apex_threshold: float = 3.5,
    lob_time_threshold: float = 1.5,
) -> dict:
    apex_values: list[float] = []
    apex_time_values: list[float] = []
    bounce_values: list[float] = []
    net_values: list[float] = []
    net_cross_values: list[float] = []
    horizontal_range_values: list[float] = []
    flight_path_values: list[float] = []
    avg_speed_values: list[float] = []
    speed_at_bounce_values: list[float] = []

    by_category: dict[str, dict[str, list[float]]] = {}
    category_counts: dict[str, int] = {}

    num_files = 0
    num_shots = 0
    rally_lengths: list[int] = []
    lob_like_by_apex = 0
    lob_like_by_time = 0

    for npz_path in _iter_npz_files(root, max_files):
        data = np.load(npz_path, allow_pickle=True)
        meta = json.loads(str(data["meta"]))
        positions = data["ball_pos_world"]
        velocities = data["ball_vel_world"]

        if "rally_length" in meta:
            rally_lengths.append(int(meta["rally_length"]))

        per_shot = _collect_shot_metrics(meta, positions, velocities)
        num_files += 1

        for shot_metrics in per_shot:
            num_shots += 1
            category = str(shot_metrics["category"])
            category_counts[category] = category_counts.get(category, 0) + 1
            by_category.setdefault(category, {
                "apex_height_m": [],
                "apex_time_s": [],
                "time_to_bounce1_s": [],
                "net_clearance_m": [],
                "net_cross_height_m": [],
                "horizontal_range_m": [],
                "flight_path_m": [],
                "avg_speed_mps": [],
                "speed_at_bounce_mps": [],
            })

            apex = shot_metrics["apex_height_m"]
            if apex is not None:
                apex_values.append(apex)
                by_category[category]["apex_height_m"].append(apex)
                if apex >= lob_apex_threshold:
                    lob_like_by_apex += 1

            apex_time = shot_metrics["apex_time_s"]
            if apex_time is not None:
                apex_time_values.append(apex_time)
                by_category[category]["apex_time_s"].append(apex_time)

            t_bounce = shot_metrics["time_to_bounce1_s"]
            if t_bounce is not None:
                bounce_values.append(t_bounce)
                by_category[category]["time_to_bounce1_s"].append(t_bounce)
                if t_bounce >= lob_time_threshold:
                    lob_like_by_time += 1

            net = shot_metrics["net_clearance_m"]
            if net is not None:
                net_values.append(net)
                by_category[category]["net_clearance_m"].append(net)

            net_cross = shot_metrics["net_cross_height_m"]
            if net_cross is not None:
                net_cross_values.append(net_cross)
                by_category[category]["net_cross_height_m"].append(net_cross)

            horizontal_range = shot_metrics["horizontal_range_m"]
            if horizontal_range is not None:
                horizontal_range_values.append(horizontal_range)
                by_category[category]["horizontal_range_m"].append(horizontal_range)

            flight_path = shot_metrics["flight_path_m"]
            if flight_path is not None:
                flight_path_values.append(flight_path)
                by_category[category]["flight_path_m"].append(flight_path)

            avg_speed = shot_metrics["avg_speed_mps"]
            if avg_speed is not None:
                avg_speed_values.append(avg_speed)
                by_category[category]["avg_speed_mps"].append(avg_speed)

            speed_at_bounce = shot_metrics["speed_at_bounce_mps"]
            if speed_at_bounce is not None:
                speed_at_bounce_values.append(speed_at_bounce)
                by_category[category]["speed_at_bounce_mps"].append(speed_at_bounce)

    results = {
        "num_files": num_files,
        "num_shots": num_shots,
        "category_counts": category_counts,
        "rally_length": _summary(rally_lengths).to_dict() if rally_lengths else None,
        "lob_like": {
            "apex_threshold_m": lob_apex_threshold,
            "time_threshold_s": lob_time_threshold,
            "by_apex_ratio": float(lob_like_by_apex) / float(num_shots)
            if num_shots > 0
            else 0.0,
            "by_time_ratio": float(lob_like_by_time) / float(num_shots)
            if num_shots > 0
            else 0.0,
        },
        "metrics": {
            "apex_height_m": _summary(apex_values).to_dict() if apex_values else None,
            "apex_time_s": _summary(apex_time_values).to_dict()
            if apex_time_values
            else None,
            "time_to_bounce1_s": _summary(bounce_values).to_dict()
            if bounce_values
            else None,
            "net_clearance_m": _summary(net_values).to_dict() if net_values else None,
            "net_cross_height_m": _summary(net_cross_values).to_dict()
            if net_cross_values
            else None,
            "horizontal_range_m": _summary(horizontal_range_values).to_dict()
            if horizontal_range_values
            else None,
            "flight_path_m": _summary(flight_path_values).to_dict()
            if flight_path_values
            else None,
            "avg_speed_mps": _summary(avg_speed_values).to_dict()
            if avg_speed_values
            else None,
            "speed_at_bounce_mps": _summary(speed_at_bounce_values).to_dict()
            if speed_at_bounce_values
            else None,
        },
        "by_category": {},
    }

    for category, values in by_category.items():
        results["by_category"][category] = {
            "apex_height_m": _summary(values["apex_height_m"]).to_dict()
            if values["apex_height_m"]
            else None,
            "apex_time_s": _summary(values["apex_time_s"]).to_dict()
            if values["apex_time_s"]
            else None,
            "time_to_bounce1_s": _summary(values["time_to_bounce1_s"]).to_dict()
            if values["time_to_bounce1_s"]
            else None,
            "net_clearance_m": _summary(values["net_clearance_m"]).to_dict()
            if values["net_clearance_m"]
            else None,
            "net_cross_height_m": _summary(values["net_cross_height_m"]).to_dict()
            if values["net_cross_height_m"]
            else None,
            "horizontal_range_m": _summary(values["horizontal_range_m"]).to_dict()
            if values["horizontal_range_m"]
            else None,
            "flight_path_m": _summary(values["flight_path_m"]).to_dict()
            if values["flight_path_m"]
            else None,
            "avg_speed_mps": _summary(values["avg_speed_mps"]).to_dict()
            if values["avg_speed_mps"]
            else None,
            "speed_at_bounce_mps": _summary(values["speed_at_bounce_mps"]).to_dict()
            if values["speed_at_bounce_mps"]
            else None,
        }

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate BLCS dataset trajectory distributions."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to dataset output directory containing .npz files.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on number of .npz files to read.",
    )
    parser.add_argument(
        "--lob-apex-threshold",
        type=float,
        default=3.5,
        help="Apex height threshold (m) for lob-like ratio.",
    )
    parser.add_argument(
        "--lob-time-threshold",
        type=float,
        default=1.5,
        help="Time-to-bounce threshold (s) for lob-like ratio.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save metrics JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.input
    if not root.exists():
        raise FileNotFoundError(f"Input path not found: {root}")

    results = evaluate_dataset(
        root=root,
        max_files=args.max_files,
        lob_apex_threshold=float(args.lob_apex_threshold),
        lob_time_threshold=float(args.lob_time_threshold),
    )
    print(json.dumps(results, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
