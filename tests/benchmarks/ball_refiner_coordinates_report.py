"""Verify the 15-condition comparison and measure isolated inference latency."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from omegaconf import OmegaConf

from src.tasks.ball_refiner.coordinates.data import normalization
from src.tasks.ball_refiner.coordinates.evaluation import error_metrics
from src.tasks.ball_refiner.coordinates.inference import load_checkpoint

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_curves(run: Path, output: Path) -> None:
    training = [json.loads(line) for line in (run / "learning_curve.jsonl").read_text().splitlines()]
    validation = [(int(path.stem.split("-")[-1]), json.loads(path.read_text())) for path in sorted(run.glob("validation-*.json"))]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].plot([row["step"] for row in training], [row["reconstruction"] for row in training])
    axes[0].set(title="Training reconstruction objective", xlabel="Update", ylabel="Training loss (model-specific)")
    for name in ("all", "missing", "event"):
        axes[1].plot([step for step, _ in validation], [row[name]["rmse"] for _, row in validation], label=name)
    axes[1].set(title="Fixed validation corruption", xlabel="Update", ylabel=f"RMSE ({validation[0][1]['unit']})")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)


def latency(checkpoint: Path, arrays: dict[str, np.ndarray], device: torch.device) -> dict[str, float]:
    model, _ = load_checkpoint(checkpoint, device)
    scale, offset = normalization(model.config.dimensions)
    # First physical rally/view, one full model window. No host/device copy in timing.
    length = model.config.window_length
    selected = (arrays["rally_id"] == arrays["rally_id"][0]) & (arrays["view_id"] == 0)
    inputs = torch.from_numpy((arrays["input"][selected][:length] / scale - offset)[None]).to(device)
    missing = torch.from_numpy(arrays["missing"][selected][:length][None]).to(device)
    generator = torch.Generator(device=device).manual_seed(42)
    samples = []
    with torch.no_grad():
        for iteration in range(25):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model.predict(inputs, missing, generator=generator)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if iteration >= 5:
                samples.append((time.perf_counter() - start) * 1000)
    return {"window_frames": length, "median_ms_per_window": float(np.median(samples)),
            "p95_ms_per_window": float(np.quantile(samples, 0.95)), "median_ms_per_frame": float(np.median(samples) / length)}


def motion_metrics(arrays: dict[str, np.ndarray], fps: float) -> dict[str, float]:
    velocity_error, acceleration_error, magnitude, reference = [], [], [], []
    for rally in np.unique(arrays["rally_id"]):
        for view in np.unique(arrays["view_id"]):
            selected = (arrays["rally_id"] == rally) & (arrays["view_id"] == view)
            prediction, target = arrays["prediction"][selected], arrays["target"][selected]
            if not np.array_equal(arrays["frame_id"][selected], np.arange(len(prediction))):
                raise ValueError("Predictions do not preserve the full ordered timeline")
            pv, tv = np.diff(prediction, axis=0) * fps, np.diff(target, axis=0) * fps
            pa, ta = np.diff(pv, axis=0) * fps, np.diff(tv, axis=0) * fps
            velocity_error.extend(np.sum((pv - tv)**2, axis=-1).tolist())
            acceleration_error.extend(np.sum((pa - ta)**2, axis=-1).tolist())
            event = arrays["event"][selected][1:-1]
            magnitude.extend(np.linalg.norm(pa[event], axis=-1).tolist())
            reference.extend(np.linalg.norm(ta[event], axis=-1).tolist())
    return {"velocity_rmse_per_s": float(np.sqrt(np.mean(velocity_error))),
            "acceleration_rmse_per_s2": float(np.sqrt(np.mean(acceleration_error))),
            "event_acceleration_magnitude_ratio": float(np.mean(magnitude) / np.mean(reference))}


def report(runs_root: Path, output: Path, *, tag: str, device: str) -> None:
    if output.exists():
        raise FileExistsError(output)
    if device == "cuda" and not os.environ.get("TENNIS_RUN_ID"):
        raise RuntimeError("CUDA timing requires the shared queue, resource=all")
    torch.set_num_threads(2)
    runs = sorted(runs_root.glob(f"coordinates-*/{tag}-s42"))
    if len(runs) != 15:
        raise ValueError(f"Expected all 15 conditions, found {len(runs)}")
    output.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    inputs_by_dimension: dict[int, str] = {}
    common_contract = None
    conditions = set()
    for run in runs:
        config = OmegaConf.to_container(OmegaConf.load(run / "config.yaml"), resolve=True)
        if not isinstance(config, dict):
            raise ValueError("Invalid config")
        state = json.loads((run / "state.json").read_text())
        if state["status"] != "complete":
            raise ValueError(f"Run is not complete: {run}")
        contract = json.loads((run / "data_contract.json").read_text())
        shared = {key: contract[key] for key in ("manifest_sha256", "train_ids", "val_ids", "test_ids", "evaluation_seed", "evaluation_corruption", "fps")}
        shared.update(steps=config["training"]["steps"], batch_size=config["training"]["batch_size"], seed=config["run"]["seed"])
        if common_contract is None:
            common_contract = shared
        elif shared != common_contract:
            raise ValueError(f"Unfair data/split/corruption/budget comparison: {run}")
        detail = json.loads((run / "predictions/diagnostic_metrics.json").read_text())
        dimensions = config["model"]["dimensions"]
        architecture = config["model"]["architecture"]
        method = "flow" if architecture == "flow" else ("gan" if config["training"]["gan_weight"] else "regression")
        rate = config["corruption"]["event_probability"]
        identity = dimensions, method, rate
        if identity in conditions:
            raise ValueError("Repeated ablation condition")
        conditions.add(identity)
        if dimensions in inputs_by_dimension and inputs_by_dimension[dimensions] != detail["evaluation_input_sha256"]:
            raise ValueError("Evaluation inputs differ between models")
        inputs_by_dimension[dimensions] = detail["evaluation_input_sha256"]
        with np.load(run / "predictions/pred_test.npz", allow_pickle=False) as values:
            arrays = {key: values[key] for key in values.files}
        repeated = error_metrics(arrays["prediction"], arrays["target"], arrays["missing"], arrays["event"])
        for stratum in ("all", "missing", "event", "observed"):
            if repeated[stratum] != detail[stratum]:
                raise ValueError(f"Saved inference does not reproduce metrics for {run}")
        row = {"dimensions": dimensions, "method": method, "event_probability": rate, "run": str(run),
               "unit": detail["unit"], "validation_rmse": detail["validation_rmse"], "best_step": detail["best_step"],
               "rmse": detail["all"]["rmse"], "missing_rmse": detail["missing"]["rmse"], "event_rmse": detail["event"]["rmse"],
               "observed_rmse": detail["observed"]["rmse"], "frame_missing_rate": detail["frame_missing_rate"],
               "linear_baseline": detail["linear_baseline"], "training_seconds": detail["training_seconds"],
               "peak_gpu_memory_bytes": detail["peak_gpu_memory_bytes"], "motion": motion_metrics(arrays, contract["fps"]),
               "latency": latency(run / "logs/version_0/checkpoints/best.ckpt", arrays, torch.device(device))}
        plot_curves(run, run / "learning_curves.png")
        rows.append(row)
        print(json.dumps({key: row[key] for key in ("dimensions", "method", "event_probability", "rmse", "latency")}), flush=True)
    expected = {(d, method, rate) for d in (2, 3) for method in (("regression", "gan") if d == 2 else ("regression", "gan", "flow")) for rate in (0.25, 0.5, 0.75)}
    if conditions != expected:
        raise ValueError("The declared ablation grid is incomplete")
    selected = {str(d): min((r for r in rows if r["dimensions"] == d), key=lambda r: r["validation_rmse"]) for d in (2, 3)}
    payload = {"common_contract": common_contract, "evaluation_input_sha256": inputs_by_dimension, "rows": rows, "selected_by_validation": selected,
               "latency_protocol": "exclusive queue; batch=1,T=128; 5 warmups,20 timed calls; sync CUDA; exclude model load and transfers",
               "scope": "synthetic fully-in-frame BLCS trajectories; seed42 only; no real-video accuracy claim"}
    (output / "comparison.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    lines = ["# 座標Refiner共通データ比較", "", "同一rally split・評価劣化・4,000更新。候補の選択はvalidation RMSEによる。", "",
             "| 次元 | 方式 | 学習イベント選択率 | test RMSE | 欠損RMSE | event RMSE | 推論ms/128frame |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['dimensions']}D | {row['method']} | {row['event_probability']:.0%} | {row['rmse']:.3f} {row['unit']} | {row['missing_rmse']:.3f} | {row['event_rmse']:.3f} | {row['latency']['median_ms_per_window']:.2f} |")
    lines += ["", "合成データ・1 seedの比較。実動画での性能とGANの一般的な優位は未確認。推論時間は単独GPU、batch=1、warmup後の中央値。"]
    (output / "comparison.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    if not args.runs_root.is_absolute() or not args.output.is_absolute():
        parser.error("Use explicit absolute roots")
    report(args.runs_root, args.output, tag=args.tag, device=args.device)


if __name__ == "__main__":
    main()
