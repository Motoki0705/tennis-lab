"""Render the recorded validation results; no model execution or new evaluation."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
best = json.loads((HERE / "best.json").read_text())
stress = json.loads((HERE / "stress.json").read_text())
parent = json.loads(
    (HERE.parent / "run-i1050-convnext-v2-pretrain-s42/best.json").read_text()
)
history = [json.loads(x) for x in (HERE / "metrics.jsonl").read_text().splitlines()]
plt.rcParams.update(
    {
        "font.family": "Noto Sans CJK JP",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#B8C2CC",
        "axes.titlelocation": "left",
        "text.color": "#172A3A",
        "axes.labelcolor": "#172A3A",
        "xtick.color": "#425466",
        "ytick.color": "#425466",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)
blue, orange, teal, purple = "#2369BD", "#D07720", "#13897F", "#8656AB"
profiles = {"clean": best, **stress}
names = {
    "clean": "通常",
    "occlusion": "人工遮蔽",
    "camera": "カメラ移動",
    "combined": "移動＋遮蔽",
}
order = ["clean", "occlusion", "camera", "combined"]
colors = {"clean": blue, "occlusion": teal, "camera": orange, "combined": purple}

fig1, axes = plt.subplots(
    1, 2, figsize=(11.69, 8.27), gridspec_kw={"width_ratios": [1.18, 1]}
)
fig1.subplots_adjust(top=0.79, bottom=0.24, left=0.075, right=0.97, wspace=0.3)
fig1.text(
    0.055, 0.935, "ConvNeXt V2 → Transformer 事後学習", fontsize=21, weight="bold"
)
fig1.text(
    0.055,
    0.883,
    "60,000更新完了  |  BF16・seed 42  |  query-only / MDD入力  |  2026-10-11",
    fontsize=11,
)
ax = axes[0]
for scope, color, label in (
    ("common", blue, "Common（56 clips）"),
    ("full", orange, "Full（190 clips）"),
):
    ax.plot(
        [r["global_step"] / 1000 for r in history],
        [r["scopes"][scope]["macro_mean_error_px"] for r in history],
        "o-",
        color=color,
        lw=2,
        ms=4,
        label=label,
    )
    ax.axhline(
        parent["scopes"][scope]["macro_mean_error_px"],
        color=color,
        ls="--",
        lw=1,
        alpha=0.6,
    )
ax.axvspan(0, 6, color="#E9EFF4", zorder=-1)
ax.text(3, 55, "CNN\n固定", ha="center", va="top", fontsize=9)
ax.text(32, 55, "CNN＋Transformer 微調整", ha="center", fontsize=10)
ax.set(
    title="① 通常validationの推移",
    xlabel="事後学習の更新数（千）",
    ylabel="平均位置誤差（元画像 px）",
    xlim=(0, 63),
    ylim=(0, 60),
)
ax.set_xticks(range(0, 61, 12))
ax.grid(axis="y", color="#E8EDF2")
ax.legend(frameon=False, loc="center right", fontsize=10)
ax.text(
    0.04,
    0.07,
    "破線：親DPTの選択checkpoint\nCommon 17.35 / Full 20.01 px",
    transform=ax.transAxes,
    fontsize=9,
    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
)
ax = axes[1]
y = np.arange(len(order))
for delta, scope, color, label in (
    (-0.16, "common", blue, "Common"),
    (0.16, "full", orange, "Full"),
):
    values = [profiles[p]["scopes"][scope]["macro_mean_error_px"] for p in order]
    bars = ax.barh(y + delta, values, height=0.29, color=color, label=label)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=10)
ax.set_yticks(y, [names[p] for p in order])
ax.invert_yaxis()
ax.set(
    title="② 同じbest checkpointの条件別評価",
    xlabel="平均位置誤差（元画像 px）",
    xlim=(0, 39),
)
ax.grid(axis="x", color="#E8EDF2")
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper right", fontsize=10)
fig1.text(
    0.055,
    0.165,
    "通常評価：Common 13.49 px / Full 15.38 px。親DPT比で22.2% / 23.1%低下。",
    fontsize=12,
    weight="bold",
)
fig1.text(
    0.055,
    0.115,
    "各FPSの平均誤差を等重みで平均。Commonはpose側とGT・split等が一致するsubsetで、このモデルはposeを入力しない。\nカメラ条件は画面外GTを除くため採点集合が僅かに異なる。単一seed・test未使用・拡張なし対照実験なし。",
    fontsize=9.5,
    linespacing=1.8,
)
fig1.text(
    0.055,
    0.052,
    "DPT → Transformer、追加学習、拡張を同時に変更しているため、個別の施策による効果は分離していない。",
    fontsize=9.5,
    color="#5C6B79",
)

fig2, axes = plt.subplots(
    1, 3, figsize=(11.69, 8.27), gridspec_kw={"width_ratios": [1.12, 1, 1]}
)
fig2.subplots_adjust(top=0.79, bottom=0.29, left=0.065, right=0.96, wspace=0.48)
fig2.text(
    0.055,
    0.935,
    "残る誤差：低FPSのカメラ移動と大きな外れ値",
    fontsize=20,
    weight="bold",
)
fig2.text(
    0.055,
    0.883,
    "すべてFull validation（190 clips）  |  選択checkpoint：epoch-009 / 60,000更新",
    fontsize=11,
)
x = np.arange(3)
labels = ["元FPS", "1/2 FPS", "1/4 FPS"]
for profile in order:
    vals = [
        profiles[profile]["scopes"]["full"]["by_frame_step"][str(s)]["mean_error_px"]
        for s in (1, 2, 4)
    ]
    axes[0].plot(x, vals, "o-", color=colors[profile], lw=2, ms=5, label=names[profile])
axes[0].set(title="③ FPS別の平均誤差", ylabel="平均位置誤差（元画像 px）", ylim=(0, 50))
axes[0].legend(frameon=False, fontsize=9, loc="upper left")
fps = best["scopes"]["full"]["by_frame_step"]
for key, color, label in (
    ("median_error_px", teal, "中央値（P50）"),
    ("mean_error_px", blue, "平均"),
    ("p95_error_px", orange, "P95"),
):
    vals = [fps[str(s)][key] for s in (1, 2, 4)]
    axes[1].plot(x, vals, "o-", color=color, lw=2, ms=5, label=label)
    for xx, val in zip(x, vals, strict=True):
        axes[1].annotate(
            f"{val:.1f}",
            (xx, val),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
axes[1].set(title="④ 通常評価の分布要約", ylabel="位置誤差（元画像 px）", ylim=(0, 58))
axes[1].legend(frameon=False, fontsize=9, loc="upper center")
for ax in axes[:2]:
    ax.set_xticks(x, labels, rotation=12)
    ax.set_xlim(-0.2, 2.2)
    ax.grid(axis="y", color="#E8EDF2")
sources = ["tracknet", "chat_annotation", "meiji"]
vals = [best["scopes"]["full"]["by_source"][s]["macro_mean_error_px"] for s in sources]
bars = axes[2].barh(range(3), vals, color=[teal, blue, orange], height=0.45)
axes[2].bar_label(bars, fmt="%.2f", padding=4, fontsize=10)
axes[2].set_yticks(range(3), ["TrackNet", "Chat", "Meiji"])
axes[2].invert_yaxis()
axes[2].set(
    title="⑤ 通常評価のデータ源別", xlabel="平均位置誤差（元画像 px）", xlim=(0, 34)
)
axes[2].grid(axis="x", color="#E8EDF2")
axes[2].set_axisbelow(True)
fig2.text(
    0.055,
    0.205,
    "カメラ移動：1/4 FPSで41.11 px。通常評価の中央値は約5–6 pxだが、P95は約39–47 px。",
    fontsize=11.5,
    weight="bold",
)
fig2.text(
    0.055,
    0.151,
    "P95：採点点の95%がこの誤差以下。データ源ごとに元画像の解像度・撮影条件・分布が異なり、\nこの図だけで原因は特定できない。人工遮蔽点だけの平均は32.94 px（全点では16.64 px）。",
    fontsize=9.5,
    linespacing=1.8,
)
fig2.text(
    0.055,
    0.077,
    "カメラ拡張は2D affine（pan / zoom / roll / jitter）。視差・新視点・scene cutは再現しない。\n条件ごとに乱数が異なるため、移動＋遮蔽が移動単独より僅かに良いことを改善の証拠とはしない。",
    fontsize=9.5,
    linespacing=1.8,
    color="#5C6B79",
)

for name, fig in (("learning-and-stress", fig1), ("fps-and-error-distribution", fig2)):
    fig.savefig(HERE / f"{name}.png", dpi=200)
with PdfPages(HERE / "validation-figures.pdf") as pdf:
    pdf.savefig(fig1)
    pdf.savefig(fig2)
plt.close("all")
print("Rendered 2 PNG figures and a 2-page PDF from saved metrics.")
