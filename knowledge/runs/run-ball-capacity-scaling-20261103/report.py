"""Analyze only completed, identity-matched validation runs for #1675."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONDITIONS = [(256, 4), (384, 4), (512, 4), (768, 4), (1024, 4), (512, 6), (512, 8)]


def read(path):
    return json.loads(path.read_text())


def summarize(campaign, theory_path, output):
    complete = read(campaign / "COMPLETED.json")
    if complete.get("complete") is not True or complete["optimizer_updates"] != 52000:
        raise ValueError(
            "All 52,000 physical updates must complete before a final accuracy report"
        )
    plan = read(campaign / "plan.json")
    bench = read(campaign / "benchmark.json")
    theory = read(theory_path)
    if (
        len(plan["stages"]) != 8
        or sum(s["budget"]["updates"] for s in plan["stages"]) != 52000
    ):
        raise ValueError(
            "Expected one common pretraining and exactly seven independent query branches"
        )
    rows = []
    identities = None
    decoder = None
    draw_digest = None
    for stage in plan["stages"]:
        name = stage["name"]
        architecture = stage["architecture"]
        root = campaign / name
        updates = stage["budget"]["updates"]
        done = read(root / "COMPLETED.json")
        if done["complete"] is not True or done["updates_completed"] != updates:
            raise ValueError("Stage is incomplete: " + name)
        report = read(root / f"validation-update-{updates:08d}.json")
        metrics = report["metrics"]
        if report["split"] != "val" or report["runtime"]["precision"] != "bf16":
            raise ValueError("Validation split/precision changed")
        if (
            metrics["support"]["observed"] != 85473
            or metrics["support"]["frames"] != 128668
        ):
            raise ValueError("Frozen validation support changed")
        if identities is None:
            identities = report["identities"]
            decoder = report["image_decode"]
        if report["identities"] != identities or report["image_decode"] != decoder:
            raise ValueError(
                "Validation owner/teacher/frame identities or decoder differ"
            )
        records = [
            json.loads(s) for s in (root / "train.jsonl").read_text().splitlines()
        ]
        if [r["update"] for r in records] != list(range(1, updates + 1)) or any(
            r["input_frames"] != 128 for r in records
        ):
            raise ValueError("Training update sequence or physical budget changed")
        if architecture == "query":
            dig = hashlib.sha256()
            for r in records:
                dig.update(
                    (
                        json.dumps(
                            [
                                r["pack_ids"],
                                r["pack_indices"],
                                r["augmentation_sha256"],
                            ],
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode()
                )
            if draw_digest is None:
                draw_digest = dig.hexdigest()
            if dig.hexdigest() != draw_digest:
                raise ValueError(
                    "Query branches did not use identical pack draws and augmentations"
                )
        config = stage["model"]
        dim = config["dim"]
        layers = config["layers"]
        count = next(
            x
            for x in theory["cases"]
            if x["architecture"] == architecture
            and (architecture == "dpt" or (x["dim"], x["layers"]) == (dim, layers))
        )
        micro = stage["budget"]["microbatch_windows"]
        measured = next(
            x
            for x in bench["cases"]
            if x["success"]
            and x["architecture"] == architecture
            and x["microbatch_windows"] == micro
            and (architecture == "dpt" or (x["dim"], x["layers"]) == (dim, layers))
        )
        joint = records[max(5, stage["budget"]["freeze_updates"] + 5) :]
        row = dict(
            stage=name,
            architecture=architecture,
            dim=dim if architecture == "query" else None,
            layers=layers if architecture == "query" else None,
            heads=config["heads"] if architecture == "query" else None,
            ffn_dim=config["ffn_dim"] if architecture == "query" else None,
            frames=config["frames"],
            updates=updates,
            mean_error_px=metrics["mean_error_px"],
            median_error_px=metrics["median_error_px"],
            p95_error_px=metrics["p95_error_px"],
            within_20px_rate=metrics["within_distance_rate"],
            total_parameters=count["total_parameters"],
            gmac_per_frame=count["total_macs_per_frame"] / 1e9,
            benchmark_median_s_per_update=measured["median_update_seconds"],
            training_joint_median_s_per_update=statistics.median(
                r["seconds"] for r in joint
            ),
            benchmark_peak_allocated_gib=measured["peak_vram_bytes"] / 2**30,
            training_peak_allocated_gib=max(r["peak_vram_bytes"] for r in records)
            / 2**30,
            training_peak_reserved_gib=max(
                r["peak_reserved_vram_bytes"] for r in records
            )
            / 2**30,
            measured_training_update_hours=sum(r["seconds"] for r in records) / 3600,
            microbatch_windows=micro,
            validation_seconds=report["seconds"],
            support=metrics["support"],
        )
        rows.append(row)
    queries = [r for r in rows if r["architecture"] == "query"]
    if {(r["dim"], r["layers"]) for r in queries} != set(CONDITIONS):
        raise ValueError("Query condition matrix differs")
    comparisons = []
    for axis, ordered in [
        (
            "dim",
            sorted((r for r in queries if r["layers"] == 4), key=lambda r: r["dim"]),
        ),
        (
            "layers",
            sorted((r for r in queries if r["dim"] == 512), key=lambda r: r["layers"]),
        ),
    ]:
        for small, large in zip(ordered, ordered[1:], strict=False):
            param_doubles = math.log2(
                large["total_parameters"] / small["total_parameters"]
            )
            mac_doubles = math.log2(large["gmac_per_frame"] / small["gmac_per_frame"])
            comparisons.append(
                dict(
                    axis=axis,
                    smaller=small["stage"],
                    larger=large["stage"],
                    parameter_ratio=large["total_parameters"]
                    / small["total_parameters"],
                    gmac_ratio=large["gmac_per_frame"] / small["gmac_per_frame"],
                    time_ratio=large["training_joint_median_s_per_update"]
                    / small["training_joint_median_s_per_update"],
                    mean_improvement_px_per_parameter_doubling=(
                        small["mean_error_px"] - large["mean_error_px"]
                    )
                    / param_doubles,
                    median_improvement_px_per_parameter_doubling=(
                        small["median_error_px"] - large["median_error_px"]
                    )
                    / param_doubles,
                    p95_improvement_px_per_parameter_doubling=(
                        small["p95_error_px"] - large["p95_error_px"]
                    )
                    / param_doubles,
                    within20_gain_percentage_points_per_parameter_doubling=100
                    * (large["within_20px_rate"] - small["within_20px_rate"])
                    / param_doubles,
                    mean_improvement_px_per_gmac_doubling=(
                        small["mean_error_px"] - large["mean_error_px"]
                    )
                    / mac_doubles,
                )
            )
    result = dict(
        schema="ball_capacity_results.v1",
        complete=True,
        physical_updates=52000,
        physical_input_frames=52000 * 128,
        seed=42,
        evaluation_split="val",
        identities=identities,
        query_draw_and_augmentation_sha256=draw_digest,
        rows=rows,
        adjacent_comparisons=comparisons,
        capacity_definition="Total trainable model parameters, including the shared CNN. Improvement per doubling = metric difference/log2(parameter ratio). Positive error improvement means lower error.",
        limitations=[
            "One seed (42) per condition; no seed variance, significance or causal reliability estimate.",
            "Validation comparison only; independent test was not evaluated.",
            "No direct superiority comparison with #1367/#1514; CNN topology differs.",
            "Mean is primary, median/P95/within20 are secondary; model adoption unchanged.",
        ],
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    fields = [k for k in rows[0] if k != "support"]
    with (output / "results.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for label, ordered in [
        (
            "Width (4 layers)",
            sorted((r for r in queries if r["layers"] == 4), key=lambda r: r["dim"]),
        ),
        (
            "Depth (dim 512)",
            sorted((r for r in queries if r["dim"] == 512), key=lambda r: r["layers"]),
        ),
    ]:
        for ax, field, scale, title in zip(
            axes,
            [
                "total_parameters",
                "gmac_per_frame",
                "training_joint_median_s_per_update",
            ],
            [1e6, 1, 1],
            [
                "Parameters (M)",
                "Forward GMAC / frame",
                "Joint training seconds / update",
            ],
            strict=True,
        ):
            ax.plot(
                [r[field] / scale for r in ordered],
                [r["mean_error_px"] for r in ordered],
                marker="o",
                label=label,
            )
            for r in ordered:
                ax.annotate(
                    f"{r['dim']}x{r['layers']}",
                    (r[field] / scale, r["mean_error_px"]),
                    fontsize=7,
                    xytext=(3, 3),
                    textcoords="offset points",
                )
            ax.set_xlabel(title)
            ax.set_ylabel("Validation mean error (source px)")
            ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("X3D/CSN + four-query mHC, T64, seed 42")
    fig.tight_layout()
    fig.savefig(output / "capacity.png", dpi=160)
    plt.close(fig)
    lines = [
        "# 容量アブレーションの結果",
        "",
        "共通CNN+DPTの10,000更新と、独立したTransformer 7条件の各6,000更新を完了。すべて固定validationで評価した。",
        "",
        "| 条件 | mean px | median px | P95 px | ≤20px | params M | GMAC/frame | 定常 s/update | peak GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in queries:
        lines.append(
            f"| dim{r['dim']}・{r['layers']}層 | {r['mean_error_px']:.3f} | {r['median_error_px']:.3f} | {r['p95_error_px']:.3f} | {r['within_20px_rate']:.2%} | {r['total_parameters'] / 1e6:.3f} | {r['gmac_per_frame']:.3f} | {r['training_joint_median_s_per_update']:.3f} | {r['training_peak_allocated_gib']:.3f} |"
        )
    lines += [
        "",
        "誤差・距離内率は85,473 observed frameが分母。全128,668 owner frameの同一性と、全query条件の抽出・拡張の一致を照合した。",
        "定常時間はCNN凍結解除後の更新の中央値。peakは学習中の最大allocated VRAM。GMACはforwardのConv/Linear/attention行列積で、正規化・活性化・MDD・backward等を含まない。",
        "",
        "容量はCNNを含む総パラメータ数で定義し、隣接条件の改善量をlog2(パラメータ比)で割った。全比較値はsummary.jsonへ保存した。",
        "",
        "seed 42の各1回であり、条件差がseedのばらつきを超えるかは判断できない。独立testは未評価。CNNが異なる過去実験との直接の優劣比較、sceneモデルの採用変更は行っていない。",
        "",
        "![容量・計算量・実測時間と位置精度](capacity.png)",
        "",
    ]
    (output / "results.md").write_text("\n".join(lines))
    print(
        json.dumps(
            dict(
                complete=True,
                rows=len(rows),
                comparisons=len(comparisons),
                output=str(output),
            )
        )
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--theory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    summarize(a.campaign, a.theory, a.output)


if __name__ == "__main__":
    main()
