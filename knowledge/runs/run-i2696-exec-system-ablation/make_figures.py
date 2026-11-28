"""Render the Japanese paper figure directly from metrics.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.axes import Axes

CASES = ("C1_serial_pt16", "C2_shared_pt16", "C3_batched_pt16", "C4_batched_trt16")
LABELS = ("C1", "C2", "C3", "C4")
CASE_COLORS = ("#8B98A8", "#246084", "#1F896F", "#A95628")
STAGES = (
    ("計測・計画", "#9AB0C8", ("measurement", "planning")),
    ("モデル読込・解放", "#594E88", ("model_load", "model_unload")),
    ("デコード", "#E6BD72", ("decode", "decode_queue_wait", "video_read")),
    ("前処理・バッチ制御", "#2B7E91", ("preprocess", "batch_dispatch")),
    ("推論", "#345B98", ("inference",)),
    ("転送・同期", "#9E5894", ("transfer", "cuda_sync")),
    ("後処理", "#1D7964", ("postprocess",)),
    (
        "成果物 I/O",
        "#C86F4F",
        ("artifact_load", "save", "publication", "progress_write"),
    ),
    ("区間外・その他", "#DFE3E8", ()),
)
MODELS = (
    ("DINO Swin-L", "dino"),
    ("ViTPose-H", "vitpose"),
    ("CLIP-ReID", "clip_reid_union"),
    ("HMR2", "hmr2"),
    ("GVHMR", "gvhmr"),
)


def plot_stage_time(axis: Axes, rows: list[dict[str, Any]]) -> None:
    cli = np.array([row["cli_wall_seconds"] for row in rows])
    left = np.zeros(len(rows))
    for label, color, stages in STAGES:
        width = (
            np.array(
                [
                    sum(row["stage_wall_seconds"].get(stage, 0) for stage in stages)
                    for row in rows
                ]
            )
            if stages
            else cli - left
        )
        axis.barh(LABELS, width, left=left, height=0.57, color=color, label=label)
        left += width
    for index, row in enumerate(rows):
        axis.annotate(
            f"{cli[index]:,.1f} 秒  /  {row['speedup_vs_c1']:.2f}×",
            (cli[index], index),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.4,
        )
    axis.invert_yaxis()
    axis.set_xlim(0, max(cli) * 1.29)
    axis.set_xlabel("経過時間 [秒]（倍率は C1 比）")
    axis.set_title("(a1) E2E 経過時間と段階別内訳", loc="left", pad=10)
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(0, -0.22),
        ncols=3,
        fontsize=8.1,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.15,
        borderaxespad=0,
    )


def plot_model_time(axis: Axes, rows: list[dict[str, Any]]) -> None:
    y = np.arange(len(MODELS))
    height = 0.17
    for index, (label, color, row) in enumerate(
        zip(LABELS, CASE_COLORS, rows, strict=True)
    ):
        values = [row["model_wall_seconds"][name] for _, name in MODELS]
        axis.barh(y + (index - 1.5) * height, values, height, color=color, label=label)
    axis.set_yticks(y, labels=[label for label, _ in MODELS])
    axis.invert_yaxis()
    axis.set_xlabel("モデル別のホスト区間和 [秒]")
    axis.set_title("(a2) モデル別の計測時間", loc="left", pad=10)
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(0, -0.22),
        ncols=4,
        fontsize=8.5,
        frameon=False,
        borderaxespad=0,
    )


def plot_memory(axis: Axes, rows: list[dict[str, Any]]) -> None:
    x = np.arange(len(rows))
    for offset, key, label, color in (
        (-0.19, "peak_vram_gib", "NVML VRAM", "#2B7E91"),
        (0.19, "peak_rss_gib", "RSS", "#594E88"),
    ):
        series = axis.bar(
            x + offset, [row[key] for row in rows], 0.34, color=color, label=label
        )
        axis.bar_label(series, fmt="%.2f", padding=3, fontsize=8.5)
    axis.set_xticks(x, LABELS)
    axis.set_ylim(0, max(row["peak_rss_gib"] for row in rows) * 1.23)
    axis.set_ylabel("使用量 [GiB]")
    axis.set_title("(b1) ピークメモリ", loc="left", pad=10)
    axis.legend(loc="upper right", ncols=2, fontsize=8.5, frameon=False)


def plot_loads(axis: Axes, rows: list[dict[str, Any]]) -> None:
    geometry = np.array([row["models"]["body_geometry"]["load_count"] for row in rows])
    loads = np.array([row["managed_model_loads"] for row in rows])
    axis.bar(LABELS, loads - geometry, width=0.5, color="#2B7E91", label="主要5モデル")
    total = axis.bar(
        LABELS,
        geometry,
        bottom=loads - geometry,
        width=0.5,
        color="#8B98A8",
        label="SMPL 身体モデル",
    )
    axis.bar_label(total, labels=[str(n) for n in loads], padding=3, fontsize=9)
    axis.set_ylim(0, max(loads) * 1.25)
    axis.set_yticks(range(0, max(loads) + 1, 3))
    axis.set_ylabel("ロード回数 [回]")
    axis.set_title("(b2) ランタイム管理のモデルロード", loc="left", pad=10)
    axis.legend(loc="upper right", fontsize=8.5, frameon=False)


def render(metrics: dict[str, Any], destination: Path) -> None:
    rows = [metrics["conditions"][case] for case in CASES]
    font_manager.findfont("Noto Sans CJK JP", fallback_to_default=False)
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "axes.labelcolor": "#273746",
            "axes.axisbelow": True,
            "grid.color": "#DCE2E8",
            "grid.linewidth": 0.60,
            "text.color": "#172D3D",
            "xtick.color": "#273746",
            "ytick.color": "#273746",
            # Emit vector glyphs for the CFF font in the Noto CJK collection.
            "pdf.fonttype": 3,
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.1), layout="constrained")
        for axis, plot, grid in zip(
            axes.flat,
            (plot_stage_time, plot_model_time, plot_memory, plot_loads),
            ("x", "x", "y", "y"),
            strict=True,
        ):
            axis.grid(axis=grid)
            axis.tick_params(length=0, pad=4)
            plot(axis, rows)
        destination.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            destination / "comparison.png",
            dpi=185,
            metadata={"Software": "tennis-lab #2696"},
        )
        fig.savefig(
            destination / "comparison.pdf",
            metadata={
                "Title": "実行システムの統制比較",
                "Creator": "tennis-lab #2696",
                "CreationDate": None,
                "ModDate": None,
            },
        )
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    render(json.loads(args.metrics.read_bytes()), args.destination)


if __name__ == "__main__":
    main()
