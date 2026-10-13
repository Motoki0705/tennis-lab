"""Render observation tables and standalone figures; make no adoption decision."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from .evaluation import save_json
from .issue1044 import ARMS


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def fit_summary(report: dict[str, Any]) -> dict[str, Any]:
    fits = [clip["fit"] for clip in report["clips"] if "fit" in clip]
    supported = [row for row in fits if not row["failed"] and row["supported_frames"]]
    count = sum(row["supported_frames"] for row in supported)
    requested = sum(row["requested_frames"] for row in fits)
    return {
        "fit_rmse_m": np.sqrt(
            sum(row["rmse_m"] ** 2 * row["supported_frames"] for row in supported)
            / count
        )
        if count
        else None,
        "fit_frame_count": count,
        "fit_requested_frames": requested,
        "fit_support_rate": count / requested if requested else None,
        "fit_clip_count": len(fits),
        "fit_failed_clips": sum(row["failed"] for row in fits),
    }


def physics_head_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Keep oracle segmentation and predicted events as separate populations."""
    events = [clip["events"] for clip in report["clips"] if "events" in clip]
    tp = sum(row["true_positive"] for row in events)
    predicted = sum(row["predicted_events"] for row in events)
    target = sum(row["target_events"] for row in events)
    result = {
        "events": {
            "true_positive": tp,
            "predicted_events": predicted,
            "target_events": target,
            "f1": 2 * tp / (predicted + target) if predicted + target else None,
            "timing_mean_s": sum(
                row["timing_mean_s"] * row["true_positive"]
                for row in events
                if row["true_positive"]
            )
            / tp
            if tp
            else None,
            "clip_count": len(events),
            "partition_failed_clips": sum(row["partition_failed"] for row in events),
        },
        "parameters": {
            key: value
            for key, value in report["metrics"].items()
            if key.startswith("parameter_")
        },
    }
    for mode in ("oracle", "predicted"):
        result[f"integrated_{mode}"] = {
            "failed_clips": sum(
                clip.get(f"{mode}_integration_failed", False)
                for clip in report["clips"]
            ),
            "metrics": report.get(f"integrated_{mode}", {}).get("metrics"),
        }
    return result


def create_report(root: Path) -> None:
    out = root / "report"
    out.mkdir(exist_ok=True)
    runs = read_json(root / "completed-training.json")
    selected = read_json(root / "additional-seed-selection.json")["selected_arm"]
    reports = {
        path.parent.name: read_json(path)
        for path in sorted((root / "evaluations").glob("*/evaluation.json"))
    }
    real = {
        path.parent.name: read_json(path)
        for path in sorted((root / "real").glob("*/evaluation.json"))
    }
    receipts = {row["name"]: row["receipt"] for row in runs}
    table = []
    keys = [
        "trajectory_rmse_m",
        "trajectory_mean_m",
        "trajectory_p95_m",
        "trajectory_count",
        "trajectory_rmse_m__missing",
        "trajectory_rmse_m__event_near",
        "velocity_head_rmse_mps",
        "position_difference_velocity_rmse_mps",
        "position_difference_acceleration_rmse_mps2",
        "rollout_position_rmse_m",
        "rollout_velocity_rmse_mps",
        "reprojection_rmse_px",
        "reprojection_evaluable_rate",
        "below_ground_rate",
    ]
    for name, report in reports.items():
        row = {
            "condition": name,
            **{key: report["metrics"].get(key) for key in keys},
            **fit_summary(report),
        }
        if name in receipts:
            row["validation_trajectory_rmse_m"] = receipts[name][
                "validation_trajectory_rmse_m"
            ]
            row["training_seconds"] = receipts[name]["elapsed_seconds"]
        table.append(row)
    fields = [
        "condition",
        *keys,
        "fit_rmse_m",
        "fit_frame_count",
        "fit_requested_frames",
        "fit_support_rate",
        "fit_clip_count",
        "fit_failed_clips",
        "validation_trajectory_rmse_m",
        "training_seconds",
    ]
    with (out / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    save_json(out / "metrics.json", table)
    strata = []
    for name, report in reports.items():
        for key, value in report["metrics"].items():
            if key.startswith("trajectory_rmse_m__"):
                label = key.split("__", 1)[1]
                strata.append(
                    {
                        "condition": name,
                        "stratum": label,
                        "rmse_m": value,
                        "frames": report["metrics"][f"trajectory__{label}_count"],
                    }
                )
    save_json(out / "strata.json", strata)
    save_json(
        out / "physics_head.json",
        {
            name: physics_head_summary(report)
            for name, report in reports.items()
            if any("events" in clip for clip in report["clips"])
        },
    )
    repeated = {}
    for key in keys:
        pairs = [
            (
                reports[f"reprojection-s{s}-u4000"]["metrics"].get(key),
                reports[f"{selected}-s{s}-u4000"]["metrics"].get(key),
            )
            for s in (42, 43, 44)
        ]
        if any(a is None or b is None for a, b in pairs):
            continue
        values = np.array(pairs, dtype=float)
        repeated[key] = {
            "control_mean": float(values[:, 0].mean()),
            "control_sample_sd": float(values[:, 0].std(ddof=1)),
            "candidate_mean": float(values[:, 1].mean()),
            "candidate_sample_sd": float(values[:, 1].std(ddof=1)),
            "paired_candidate_minus_control": (values[:, 1] - values[:, 0]).tolist(),
            "seed_count": 3,
        }
    save_json(out / "repeated_seeds.json", {"candidate": selected, "metrics": repeated})
    save_json(
        out / "real_metrics.json",
        {
            name: {
                "direct": data["direct"],
                "integrated_predicted": data.get("integrated_predicted"),
                "limitations": data["limitations"],
            }
            for name, data in real.items()
        },
    )
    curves = {}
    for name in receipts:
        accumulator = EventAccumulator(
            str(root / name / "logs/version_0"), size_guidance={"scalars": 0}
        )
        accumulator.Reload()
        curves[name] = {
            tag: [
                {"step": e.step, "value": e.value, "wall_time": e.wall_time}
                for e in accumulator.Scalars(tag)
            ]
            for tag in accumulator.Tags()["scalars"]
            if tag.startswith(("gradient/", "train/loss", "val/trajectory", "lr"))
        }
    save_json(out / "training_curves.json", curves)
    _plots(out, reports, real, root, curves)
    lines = [
        "# Issue1044 実験観測表",
        "",
        "10 run、各4,000更新。checkpoint・追加seed候補はvalidationのみで選定。採用判断は未実施。",
        "",
        f"追加seed対象: `{selected}`。testは全scene・全frame、非重複128frame窓。",
        "",
        "| 条件 | 3D RMSE m | 位置差分速度 m/s | rollout位置 m | 再投影 px | frame数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    def fmt(value: Any) -> str:
        return "未定義" if value is None else f"{value:.5g}"

    for row in table:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["condition"],
                    *(
                        fmt(row[key])
                        for key in (
                            "trajectory_rmse_m",
                            "position_difference_velocity_rmse_mps",
                            "rollout_position_rmse_m",
                            "reprojection_rmse_px",
                            "trajectory_count",
                        )
                    ),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "3 seedの平均・標本SD・seedごとの差は repeated_seeds.json、各層の支持数は strata.json。Bのパラメータ・イベント・GT区間／予測区間別の積分結果は physics_head.json。",
        "",
        "既存9,000更新checkpointはhead構成と予算が異なる参考対照。幾何補間の支持不足clipは個別評価JSONに記録。GT物理を使うrolloutと、実観測の未知イベントを跨ぎ得る固定窓fitを混同しない。実観測には独立3D GTがなく、改善を実3D精度の向上とは解釈しない。",
        "",
        "数値・層別表・代表軌道・勾配を含む考察はknowledgeノードに記録する。",
    ]
    (out / "observations.md").write_text("\n".join(lines) + "\n")


def _plots(
    out: Path,
    reports: dict[str, Any],
    real: dict[str, Any],
    root: Path,
    curves: dict[str, Any],
) -> None:
    names = [f"{arm}-s42-u4000" for arm in ARMS]
    labels = [
        "xyz",
        "+reproj",
        "+velocity",
        "+rollout p",
        "+rollout v (A)",
        "+params (B)",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, key, title in zip(
        axes.flat,
        (
            "trajectory_rmse_m",
            "position_difference_velocity_rmse_mps",
            "rollout_position_rmse_m",
            "reprojection_rmse_px",
        ),
        (
            "3D trajectory RMSE [m]",
            "Position-difference velocity RMSE [m/s]",
            "Event-safe rollout position residual [m]",
            "Clean reprojection RMSE [px]",
        ),
        strict=True,
    ):
        axis.bar(labels, [reports[name]["metrics"][key] for name in names])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.set_ylim(bottom=0)
    fig.savefig(out / "progressive_losses.png", dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for name, label in zip(names, labels, strict=True):
        data = curves[name]["val/trajectory_rmse_m"]
        axis.plot([x["step"] for x in data], [x["value"] for x in data], label=label)
    axis.set(
        xlabel="Optimizer updates",
        ylabel="Validation trajectory RMSE [m]",
        yscale="log",
    )
    axis.legend()
    fig.savefig(out / "validation_learning.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, name, label in zip(axes.flat, names, labels, strict=True):
        for tag, series in curves[name].items():
            if not tag.startswith("gradient/") or not tag.endswith("_weighted_l2"):
                continue
            positive = [
                row for row in series if row["value"] is not None and row["value"] > 0
            ]
            if positive:
                axis.plot(
                    [row["step"] for row in positive],
                    [row["value"] for row in positive],
                    label=tag.removeprefix("gradient/").removesuffix("_weighted_l2"),
                )
        axis.set(
            title=label,
            xlabel="Optimizer updates",
            ylabel="Output gradient L2",
            yscale="log",
        )
        axis.legend(fontsize=6)
    fig.savefig(out / "weighted_output_gradients.png", dpi=180)
    plt.close(fig)
    # Representative clips are chosen from control test error only, for display.
    control = reports["reprojection-s42-u4000"]
    clips = sorted(
        control["clips"], key=lambda row: row["metrics"]["trajectory_rmse_m"]
    )
    shown = [
        clips[0]["scene_id"],
        clips[len(clips) // 2]["scene_id"],
        clips[-1]["scene_id"],
    ]
    save_json(
        out / "display_selection.json",
        {
            "criterion": "control test clip RMSE minimum/median/maximum; visualization only",
            "scene_ids": shown,
        },
    )
    selected_names = [
        "reprojection-s42-u4000",
        "rollout_velocity-s42-u4000",
        "parameters-s42-u4000",
        "geometry",
    ]
    bundles = {
        name: np.load(root / "evaluations" / name / "pred_test.npz", allow_pickle=False)
        for name in selected_names
    }
    for index, scene in enumerate(shown):
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), constrained_layout=True)
        for name, data in bundles.items():
            matches = np.flatnonzero(data["scene_ids"] == scene)
            if not len(matches):
                continue
            row = int(matches[0])
            mask = data["frame_index"][row] >= 0
            dt = data["physics_clock"][row, 0] * data["physics_clock"][row, 3]
            t = np.arange(int(mask.sum())) * dt
            for coordinate, axis in enumerate(axes):
                axis.plot(
                    t, data["pred_position"][row, mask, coordinate] * 11.885, label=name
                )
                if name == selected_names[0]:
                    axis.plot(
                        t,
                        data["position_3d"][row, mask, coordinate] * 11.885,
                        color="black",
                        linestyle="--",
                        label="GT",
                    )
                axis.set_ylabel("xyz"[coordinate] + " [m]")
        axes[0].set_title(scene)
        axes[0].legend(fontsize=8)
        axes[-1].set_xlabel("Time [s]")
        fig.savefig(out / f"trajectory_{index}.png", dpi=160)
        plt.close(fig)
    for data in bundles.values():
        data.close()
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), constrained_layout=True)
    for name in selected_names:
        with np.load(
            root / "real" / name / "real_predictions.npz", allow_pickle=False
        ) as data:
            t = np.arange(len(data["position_m"])) / real[name]["evaluation_fps"]
            for k, axis in enumerate(axes):
                axis.plot(t, data["position_m"][:, k], label=name)
                axis.set_ylabel("xyz"[k] + " [m]")
    axes[0].set_title("Meiji observations: no independent 3D ground truth")
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("Time [s]")
    fig.savefig(out / "real_trajectories.png", dpi=160)
    plt.close(fig)
